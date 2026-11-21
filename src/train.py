"""LoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct on Spider train. Used by notebook 02.

    python -m src.train --out /kaggle/working/train                 # full run (2 epochs)
    python -m src.train --out /tmp/t --max-steps 30                 # smoke run
    python -m src.train --out /tmp/t --model <tiny model> --max-steps 2 --n-train 16   # CPU test

Outputs in --out: adapter/ (LoRA weights + config), checkpoints/, loss_history.csv, loss_curve.png,
train_info.json (settings, data sizes, timings, peak memory).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from .data import db_path, find_spider_root, load_spider, split_train_val
from .generation import BASE_MODEL
from .prompts import build_messages
from .schema import get_schema_ddl


def make_examples(examples, root, tok, max_length):
    """Prompt-completion records (TRL format). Over-long examples are dropped, not truncated,
    because truncation would cut off (part of) the SQL target."""
    records, dropped = [], 0
    for e in examples:
        msgs = build_messages(get_schema_ddl(str(db_path(root, e["db_id"]))), e["question"], e["query"])
        n_tokens = len(tok(tok.apply_chat_template(msgs, tokenize=False))["input_ids"])
        if n_tokens > max_length:
            dropped += 1
            continue
        # prompt = system + user turns (exactly the eval prompt); completion = assistant turn (gold SQL)
        records.append({"prompt": msgs[:2], "completion": msgs[2:]})
    return records, dropped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--spider-root", default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke test)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=None, help="subsample train (tests only)")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --out")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.spider_root) if args.spider_root else find_spider_root()
    cuda = torch.cuda.is_available()

    # ---- data: DB-disjoint train/val split of Spider train (dev/test is never loaded here) ----
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)  # always the Qwen tokenizer / chat template
    train_all, val_all, val_dbs = split_train_val(load_spider("train"), val_frac=0.05, seed=42)
    if args.n_train:
        train_all, val_all = train_all[: args.n_train], val_all[: max(4, args.n_train // 4)]
    train_recs, drop_tr = make_examples(train_all, root, tok, args.max_length)
    val_recs, drop_va = make_examples(val_all, root, tok, args.max_length)
    print(f"train {len(train_recs)} (dropped {drop_tr} > {args.max_length} tokens), "
          f"val {len(val_recs)} (dropped {drop_va}), val DBs: {val_dbs}")

    # ---- model: frozen base in fp16 on GPU (T4: no bf16); LoRA weights are kept in fp32 by PEFT ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if cuda else torch.float32, attn_implementation="sdpa")
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
        max_grad_norm=1.0, seed=42,
        fp16=cuda, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,  # loss only on the SQL completion, prompt tokens are masked (-100)
        max_length=args.max_length,
        logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        report_to=[], use_cpu=not cuda, dataloader_num_workers=2 if cuda else 0,
    )

    # ---- callback that records every train/eval loss for the loss curve ----
    history = []

    class LossLogger(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **kw):
            if logs and ("loss" in logs or "eval_loss" in logs):
                if history and history[-1]["step"] == state.global_step and "eval_loss" in logs \
                        and history[-1]["eval_loss"] == logs["eval_loss"]:
                    return  # the same evaluation logged twice (end of training)
                history.append({"step": state.global_step, "epoch": round(state.epoch or 0, 3),
                                "train_loss": logs.get("loss"), "eval_loss": logs.get("eval_loss"),
                                "lr": logs.get("learning_rate")})
                print(history[-1], flush=True)

    trainer = SFTTrainer(
        model=model, args=cfg, processing_class=tok, peft_config=lora,
        train_dataset=Dataset.from_list(train_recs), eval_dataset=Dataset.from_list(val_recs),
        callbacks=[LossLogger()],
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    last_ckpt = None
    if args.resume and (out / "checkpoints").exists():
        ckpts = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        last_ckpt = str(ckpts[-1]) if ckpts else None
    result = trainer.train(resume_from_checkpoint=last_ckpt)
    train_seconds = time.time() - t0
    final_eval = trainer.evaluate()  # logged into `history` by the callback

    # ---- save adapter + logs ----
    trainer.model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    with open(out / "loss_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "eval_loss", "lr"])
        w.writeheader()
        w.writerows(history)
    plot_loss(history, out / "loss_curve.png")

    steps = trainer.state.global_step
    info = {
        "base_model": args.model, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": sorted(lora.target_modules)},
        "epochs": args.epochs, "max_steps": args.max_steps, "steps_done": steps, "lr": args.lr,
        "effective_batch": args.batch_size * args.grad_accum, "max_length": args.max_length,
        "n_train": len(train_recs), "n_val": len(val_recs), "dropped_train": drop_tr, "dropped_val": drop_va,
        "val_dbs": val_dbs, "trainable_params": trainable, "total_params": total,
        "train_seconds": round(train_seconds), "seconds_per_step": round(train_seconds / max(steps, 1), 2),
        "final_train_loss": result.training_loss, "final_eval_loss": final_eval["eval_loss"],
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
    }
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def plot_loss(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [(h["step"], h["train_loss"]) for h in history if h["train_loss"] is not None]
    ev = [(h["step"], h["eval_loss"]) for h in history if h["eval_loss"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train loss", color="#2a6fdb", linewidth=1.5)
    if ev:
        ax.plot(*zip(*ev), label="validation loss (held-out DBs)", color="#e07b00", marker="o", linewidth=1.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss (SQL tokens only)")
    ax.set_title("LoRA fine-tuning: training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

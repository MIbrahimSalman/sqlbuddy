"""End-to-end evaluation on the Spider dev set (our test set). Used by notebooks 01 and 03.

    python -m src.run_eval --out results/baseline                     # base model
    python -m src.run_eval --out results/finetuned --adapter <path>   # base + LoRA adapter
    python -m src.run_eval --out /tmp/x --n 50                        # first 50 examples only

Writes to --out: predictions.csv (one row per example), metrics.csv (overall + per hardness),
run_info.json (settings, timings, library versions).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

from .data import db_path, find_spider_root, load_spider
from .evaluate import SpiderScorer, aggregate
from .prompts import build_prompt
from .schema import get_schema_ddl
from .sql_extract import extract_sql


def write_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder or HF repo id")
    ap.add_argument("--n", type=int, default=None, help="only the first N dev examples")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--spider-root", default=None)
    args = ap.parse_args(argv)

    import torch, transformers
    from .generation import load_model, generate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.spider_root) if args.spider_root else find_spider_root()
    examples = load_spider("test")[: args.n]
    print(f"{len(examples)} test examples, spider root = {root}")

    # 1) Prompts (identical for base and fine-tuned models).
    t0 = time.time()
    tok, model = load_model(args.adapter)
    prompts = [build_prompt(tok, get_schema_ddl(str(db_path(root, e["db_id"]))), e["question"]) for e in examples]
    t_load = time.time() - t0

    # 2) Batched greedy generation.
    t0 = time.time()
    raw = generate(tok, model, prompts, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    t_gen = time.time() - t0
    print(f"generation: {t_gen:.0f}s ({t_gen / len(examples):.2f}s/example)")

    # 3) Extract SQL and score.
    t0 = time.time()
    scorer = SpiderScorer(root)
    rows = []
    for e, r in zip(examples, raw):
        pred = extract_sql(r)
        rows.append({"id": e["id"], "db_id": e["db_id"], "question": e["question"], "gold_sql": e["query"],
                     "pred_sql": pred, **scorer.score(e, pred), "raw_output": r})
    t_score = time.time() - t0

    metrics = aggregate(rows)
    write_csv(rows, out / "predictions.csv")
    write_csv(metrics, out / "metrics.csv")
    info = {
        "model": "base" if not args.adapter else f"base+lora({args.adapter})",
        "n_examples": len(examples), "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy", "seconds": {"load": round(t_load), "generate": round(t_gen), "score": round(t_score)},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "metrics": metrics,
    }
    try:
        import peft
        info["versions"]["peft"] = peft.__version__
    except ImportError:
        pass
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    for m in metrics:
        print(m)


if __name__ == "__main__":
    main()

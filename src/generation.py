"""Loading the model (base, or base + LoRA adapter) and batched greedy generation."""
from __future__ import annotations

import torch

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_model(adapter: str | None = None, device: str | None = None, dtype=None):
    """Load tokenizer + model. `adapter` = local folder or HF repo id of a LoRA adapter (or None).

    GPU -> fp16 (T4 has no bf16). CPU -> fp32 (fp16 matmuls are slow/unsupported on CPU).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"  # decoder-only models must be left-padded for batched generation
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # merging the LoRA weights makes inference as fast as the base
    model.eval()
    return tok, model


@torch.inference_mode()
def generate(tok, model, prompts: list[str], batch_size: int = 16, max_new_tokens: int = 256,
             progress: bool = True) -> list[str]:
    """Greedy decoding for many prompts. Prompts are sorted by length so batches have little padding;
    outputs are returned in the original order."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    outputs: list[str] = [""] * len(prompts)
    batches = range(0, len(order), batch_size)
    if progress:
        from tqdm.auto import tqdm
        batches = tqdm(batches, desc="generating")
    for start in batches:
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]  # drop the prompt part
        for i, text in zip(idx, tok.batch_decode(new_tokens, skip_special_tokens=True)):
            outputs[i] = text
    return outputs

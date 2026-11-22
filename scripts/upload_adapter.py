"""Upload the trained LoRA adapter to the Hugging Face Hub (private repo) using $HF_TOKEN.

    python scripts/upload_adapter.py <local adapter dir> [--train-info train_info.json] [--loss-png loss_curve.png]

The token is read from the environment by huggingface_hub and never printed.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

REPO_NAME = "sqlbuddy-qwen2.5-coder-1.5b-lora"

CARD = """---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
tags: [text-to-sql, spider, lora, sqlite]
---
# SQLBuddy - Qwen2.5-Coder-1.5B-Instruct LoRA for text-to-SQL (Spider)

LoRA adapter (r=16, alpha=32, dropout 0.05, attention + MLP projections) trained for 2 epochs on
Spider train (6,648 examples; 352 from 8 held-out databases used for validation), lr 2e-4 cosine,
effective batch 16, max length 1536, fp16 on one Tesla T4. Loss on the SQL completion only.

Prompt: system message + `Database schema:\\n<CREATE TABLE ...>\\n\\nQuestion: <question>` via the Qwen chat
template; the model answers with one SQLite query. Code and evaluation: SQLBuddy course project.

## Training summary
```json
{info}
```

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model = PeftModel.from_pretrained(model, "{repo}").merge_and_unload()
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--train-info", default=None)
    ap.add_argument("--loss-png", default=None)
    a = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{REPO_NAME}"
    api.create_repo(repo, private=True, exist_ok=True)

    info = json.loads(Path(a.train_info).read_text()) if a.train_info else {}
    Path(a.adapter_dir, "README.md").write_text(CARD.replace("{info}", json.dumps(info, indent=2)).replace("{repo}", repo))
    api.upload_folder(repo_id=repo, folder_path=a.adapter_dir, commit_message="Upload SQLBuddy LoRA adapter")
    if a.loss_png:
        api.upload_file(repo_id=repo, path_or_fileobj=a.loss_png, path_in_repo="loss_curve.png")
    print("uploaded to https://huggingface.co/" + repo, "(private)")


if __name__ == "__main__":
    main()

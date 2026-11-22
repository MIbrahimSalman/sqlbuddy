"""Generate the Kaggle notebooks (.ipynb) + kernel-metadata.json from the cell sources below.

Keeping notebooks as Python source here makes them easy to review and diff.
    python notebooks/build_notebooks.py                 # all notebooks, full settings
    python notebooks/build_notebooks.py --smoke 01      # notebook 01 with N_EXAMPLES=50
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
# Kaggle owner slug = username without dots ("m.name" -> "mname").
USER = os.environ.get("KAGGLE_OWNER") or os.environ.get("KAGGLE_USERNAME", "mibrahimsalman022").replace(".", "")
SPIDER_DATASET = "jeromeblanchet/yale-universitys-spider-10-nlp-dataset"
SRC_DATASET = f"{USER}/sqlbuddy-src"

# Exact library versions used on Kaggle (torch comes preinstalled in the Kaggle image).
PIP = ("%pip install -q transformers==5.22.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 "
       "huggingface_hub==2.4.0 nltk==3.9.2")
# The Kaggle image ships torchao 0.10, which PEFT 0.21 rejects ("incompatible version of torchao")
# as soon as LoRA layers are created. We don't use torchao, so remove it for notebooks that use PEFT.
NO_TORCHAO = "%pip uninstall -y -q torchao\n"
PIP_TRAIN = NO_TORCHAO + PIP + " trl==1.17.1 matplotlib==3.11.2"
PIP_PEFT = NO_TORCHAO + PIP

SETUP = r'''
# Locate our shared code (Kaggle dataset "sqlbuddy-src") and the Spider databases, then import.
import glob, os, shutil, sys, zipfile
work = "/kaggle/working/sqlbuddy"
# The package folder is the one containing run_eval.py + spider_eval/ (Kaggle may unpack it at the
# dataset root or keep a src/ folder); copy it to /kaggle/working/sqlbuddy/src.
hits = [h for h in glob.glob("/kaggle/input/**/run_eval.py", recursive=True)
        if os.path.isdir(os.path.join(os.path.dirname(h), "spider_eval"))]
if hits:
    shutil.copytree(os.path.dirname(hits[0]), work + "/src", dirs_exist_ok=True)
else:  # or it may still be a zip archive
    zips = glob.glob("/kaggle/input/**/src.zip", recursive=True)
    assert zips, "sqlbuddy-src dataset not attached"
    zipfile.ZipFile(zips[0]).extractall(work + "/src")
sys.path.insert(0, work)
print("src commit:", open(work + "/src/GIT_COMMIT").read().strip() if os.path.exists(work + "/src/GIT_COMMIT") else "?")

from src.data import find_spider_root
SPIDER_ROOT = find_spider_root("/kaggle/input")
print("Spider root:", SPIDER_ROOT)

import torch, transformers, peft, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| datasets", datasets.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip()}


def write(slug, title, cells, datasets, gpu=True, kernels=()):
    folder = HERE / slug
    folder.mkdir(exist_ok=True)
    for i, c in enumerate(cells):  # nbformat >= 4.5 wants a unique id per cell
        c["id"] = f"cell-{i:02d}"
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (folder / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": f"{USER}/sqlbuddy-{slug.replace('_', '-')}", "title": f"sqlbuddy-{slug.replace('_', '-')}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true" if gpu else "false", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": datasets, "competition_sources": [],
        "kernel_sources": list(kernels), "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", folder)


def nb01(smoke: bool):
    n = 50 if smoke else None
    return [
        md("""
# SQLBuddy 01 - Baseline evaluation (base model, no fine-tuning)
Runs `Qwen/Qwen2.5-Coder-1.5B-Instruct` **without** fine-tuning on the Spider dev set (our held-out test set,
1,034 questions). Prompt = CREATE TABLE schema + question; greedy decoding, max 256 new tokens.
Metrics: execution accuracy, official Spider exact match, valid-SQL rate, per difficulty level.
All logic lives in the shared, unit-tested `src/` package (repo: `src/run_eval.py`).
"""),
        code(PIP),
        code(SETUP),
        code(f"""
# N_EXAMPLES = None -> full dev set; 50 -> quick smoke test to measure run time.
N_EXAMPLES = {n!r}
from src.run_eval import main
args = ["--out", "/kaggle/working/baseline", "--batch-size", "16", "--max-new-tokens", "256",
        "--spider-root", str(SPIDER_ROOT)]
if N_EXAMPLES:
    args += ["--n", str(N_EXAMPLES)]
main(args)
"""),
        code("""
import pandas as pd
display(pd.read_csv("/kaggle/working/baseline/metrics.csv"))
pred = pd.read_csv("/kaggle/working/baseline/predictions.csv")
display(pred[["question", "gold_sql", "pred_sql", "hardness", "exec_match", "exact_match", "valid_sql"]].head(15))
"""),
    ]


def nb02(smoke: bool):
    steps = 30 if smoke else -1
    return [
        md("""
# SQLBuddy 02 - LoRA fine-tuning
Fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` with LoRA (r=16, alpha=32, dropout 0.05, attention + MLP
projections) on Spider train (6,648 examples; 352 from 8 held-out databases for validation).
2 epochs, lr 2e-4, cosine, effective batch 16, max length 1536, fp16, loss on the SQL completion only.
The prompt is exactly the one used for evaluation. Code: `src/train.py` in the repo.
"""),
        code("import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use one T4 (no DataParallel)\n"
             "os.environ['TOKENIZERS_PARALLELISM'] = 'false'"),
        code(PIP_TRAIN),
        code(SETUP),
        code(f"""
# MAX_STEPS = -1 -> full 2-epoch run; 30 -> smoke test to measure time/memory.
MAX_STEPS = {steps}
from src.train import main
args = ["--out", "/kaggle/working/train", "--spider-root", str(SPIDER_ROOT), "--max-steps", str(MAX_STEPS)]
if MAX_STEPS > 0:
    args += ["--eval-steps", "30"]
main(args)
"""),
        code("""
import pandas as pd
from IPython.display import Image, display
display(Image("/kaggle/working/train/loss_curve.png"))
display(pd.read_csv("/kaggle/working/train/loss_history.csv").tail(20))
"""),
    ]


def nb03(smoke: bool):
    n = 50 if smoke else None
    return [
        md("""
# SQLBuddy 03 - Fine-tuned model evaluation
Same prompts, same greedy decoding (max 256 new tokens), same metrics and the same Spider dev set as
notebook 01 - the only difference is the LoRA adapter from notebook 02 (attached as kernel input),
merged into the base model.
"""),
        code(PIP_PEFT),
        code(SETUP),
        code("""
# The adapter = output folder 'train/adapter' of the 02_train_lora kernel.
ADAPTER = os.path.dirname([p for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
                           if "checkpoint" not in p][0])
print("adapter:", ADAPTER, os.listdir(ADAPTER))
"""),
        code(f"""
N_EXAMPLES = {n!r}
from src.run_eval import main
args = ["--out", "/kaggle/working/finetuned", "--adapter", ADAPTER, "--batch-size", "16",
        "--max-new-tokens", "256", "--spider-root", str(SPIDER_ROOT)]
if N_EXAMPLES:
    args += ["--n", str(N_EXAMPLES)]
main(args)
"""),
        code("""
import pandas as pd
display(pd.read_csv("/kaggle/working/finetuned/metrics.csv"))
pred = pd.read_csv("/kaggle/working/finetuned/predictions.csv")
display(pred[["question", "gold_sql", "pred_sql", "hardness", "exec_match", "exact_match", "valid_sql"]].head(15))
"""),
    ]


def nb04(smoke: bool = False):
    """GPU demo. The app code is downloaded from GitHub (branch main) at the start of every run,
    so Kaggle always runs the latest merged app; the adapter comes from the public HF model repo.
    Run length: started from the Kaggle editor ("Run all") = 90 min demo; pushed from the CLI
    (a batch run, used only to test the notebook) = self-test + 1 min. `smoke` is kept for CLI
    compatibility and has no effect."""
    return [
        md("""
# SQLBuddy 04 - Live demo on a Kaggle GPU (fast backup for the presentation)
Runs **the latest SQLBuddy app from GitHub** (the code is downloaded from the `main` branch at the
start of every run, so every merged design change shows up here automatically; the fine-tuned adapter
comes from the public Hugging Face model repo) on a T4 GPU: answers take ~1-4 s instead of 3-15 s on a laptop CPU.

**Use:** open this notebook in the Kaggle editor -> right panel: *Accelerator = GPU T4*, *Internet = on*
-> **Run all** -> after ~3-5 min the last cell prints `PUBLIC LINK: https://....gradio.live`.
The link works for anyone for `DEMO_MINUTES` (90). Press *Stop* when you are done to save GPU quota.
"""),
        code(PIP_PEFT + " gradio==6.31.0"),
        code(r'''
# Download the latest app from GitHub (branch main): the single source of truth for the demo code.
import io, json, os, shutil, sys, urllib.request, zipfile
REPO, BRANCH = "MIbrahimSalman/sqlbuddy", "main"
sha = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"))["sha"]
data = urllib.request.urlopen(f"https://github.com/{REPO}/archive/{sha}.zip").read()
shutil.rmtree("/kaggle/working/sqlbuddy", ignore_errors=True)
zf = zipfile.ZipFile(io.BytesIO(data))
zf.extractall("/kaggle/working")
work = os.path.join("/kaggle/working", zf.namelist()[0].rstrip("/"))   # sqlbuddy-<sha>/
os.rename(work, "/kaggle/working/sqlbuddy"); work = "/kaggle/working/sqlbuddy"
print("app version: github.com/%s commit %s" % (REPO, sha[:7]))
sys.path.insert(0, os.path.join(work, "app"))

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (enable GPU T4!)")
import app as sqlbuddy_app
sqlbuddy_app.MODELS.load()                 # base model + public LoRA adapter, on the GPU
sqlbuddy_app.MODELS.warm_up(sqlbuddy_app.schema_of(sqlbuddy_app.SAMPLE_DB))   # cache the sample schema prompt
print("device:", sqlbuddy_app.MODELS.describe(), "| adapter loaded:", sqlbuddy_app.MODELS.has_adapter,
      sqlbuddy_app.MODELS.adapter_error or "")
'''),
        code("""
# Self-test: one example question through both models (same code path as the UI button)
q = "What is the average score in each course? Show the course title."
db = str(sqlbuddy_app.SAMPLE_DB)
for ft in (False, True):
    sql, df, status = sqlbuddy_app.answer_one(db, q, ft)
    print(status)
    print("  ", sql)
    print(df.head(3))
    print()
"""),
        code("""
import time
# Started from the editor ("Interactive") -> 90 min demo; pushed from the CLI ("Batch") -> quick check.
RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
DEMO_MINUTES = 1 if RUN_TYPE == "Batch" else 90
print("run type:", RUN_TYPE, "| demo minutes:", DEMO_MINUTES)
demo = sqlbuddy_app.build_ui()
demo.queue(default_concurrency_limit=8).launch(share=True, prevent_thread_lock=True, **sqlbuddy_app.LAUNCH_KW)
print("PUBLIC LINK:", demo.share_url)      # open this *.gradio.live URL in any browser
time.sleep(DEMO_MINUTES * 60)
demo.close()
"""),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", nargs="*", default=[], help="notebook numbers to build in smoke-test mode")
    a = ap.parse_args()
    write("01_baseline_eval", "Baseline eval", nb01("01" in a.smoke), [SPIDER_DATASET, SRC_DATASET])
    write("02_train_lora", "Train LoRA", nb02("02" in a.smoke), [SPIDER_DATASET, SRC_DATASET])
    write("03_finetuned_eval", "Fine-tuned eval", nb03("03" in a.smoke), [SPIDER_DATASET, SRC_DATASET],
          kernels=[f"{USER}/sqlbuddy-02-train-lora"])
    write("04_demo", "Demo", nb04(), [])  # no inputs: app comes from the HF Space, adapter from HF

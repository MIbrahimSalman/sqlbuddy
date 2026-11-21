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
PIP = ("pip install -q transformers==5.22.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 "
       "huggingface_hub==2.4.0 nltk==3.9.2")
PIP_TRAIN = PIP + " trl==1.17.1"

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


def write(slug, title, cells, datasets, gpu=True):
    folder = HERE / slug
    folder.mkdir(exist_ok=True)
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
        "kernel_sources": [], "model_sources": [],
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", nargs="*", default=[], help="notebook numbers to build in smoke-test mode")
    a = ap.parse_args()
    write("01_baseline_eval", "Baseline eval", nb01("01" in a.smoke), [SPIDER_DATASET, SRC_DATASET])

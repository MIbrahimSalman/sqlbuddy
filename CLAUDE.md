# SQLBuddy — project brief

**Start of every session:** read this file, then `docs/PROGRESS.md` (newest entry on top, with "Next steps"), then the relevant phase in `docs/PLAN.md`.

## Goal
University NLP course project: fine-tune a small LLM to translate English questions into SQL and **prove with a held-out test set** whether fine-tuning beats the base model. Grading focuses on NLP methodology and results; the demo is secondary.

## Tech choices (details + reasons in `docs/DECISIONS.md`)
- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Method: LoRA (PEFT) + TRL `SFTTrainer`, fp16 on a Kaggle T4 (no bf16 on T4). Loss on SQL completion only.
- Data: Spider 1.0. Train = `train_spider.json` (7,000) minus a small validation split; **test = dev (1,034), never trained on**.
- Spider SQLite DBs: Kaggle dataset `jeromeblanchet/yale-universitys-spider-10-nlp-dataset` (original release, contains `spider/database/`), attached as kernel input.
- Prompt: schema as canonical CREATE TABLE statements (from PRAGMA metadata) + question → SQL only. Identical prompt for base and fine-tuned.
- Status: all phases (0–8) done. Phase 8 = demo hosted on Kaggle notebook 04 (pulls the app from GitHub main every run; HF Spaces need PRO), public adapter, `scripts/deploy.sh`. Phase 7 = demo redesign, latency study (docs/PERFORMANCE.md), per-DB + out-of-domain eval (OOD 36.7 → 73.3), slide deck. Dev (1,034): base EX 56.96 / EM 32.40 / valid 76.60 → fine-tuned EX 73.11 / EM 67.99 / valid 91.78 (McNemar p = 2.4e-25). See `results/`. Adapter on HF, logs in `results/training/`.
- Decoding: greedy, `max_new_tokens=256`, batched, then SQL extraction post-processing.
- Metrics: execution accuracy (primary), exact match (Spider official exact-set-match), valid-SQL rate; all by Spider hardness (easy/medium/hard/extra) via vendored official `evaluation.py`.
- GPU work runs only on Kaggle (pushed with the `kaggle` CLI). Adapter on HF Hub `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora` (public since Phase 8).

## Repo structure
```
notebooks/   Kaggle notebooks, one folder each with kernel-metadata.json
src/         shared code: schema/prompt building, SQL extraction, evaluation (tested on CPU here)
tests/       pytest CPU tests (SPIDER_DIR=... pytest -q tests)
scripts/     deploy.sh (one command after app changes), upload_src.sh (src/ -> Kaggle dataset sqlbuddy-src)
app/         Gradio demo (requirements-demo.txt for laptops)
results/     metrics CSVs, predictions, charts, error analysis
sample_db/   small university SQLite DB + generator script (demo)
docs/        PLAN.md, PROGRESS.md, DECISIONS.md, KAGGLE_RUNS.md
```

## Conventions
- Never print, log or commit tokens (KAGGLE_API_TOKEN, HF_TOKEN). No weights or Spider data in git (see `.gitignore`).
- Pin library versions in notebooks and `requirements.txt`.
- Test `src/` on CPU with tiny samples before any Kaggle push. On a failed Kaggle run: download log, fix cause, rerun — no blind retries.
- Plan each phase in `docs/PLAN.md` and get the user's OK before implementing it.
- Update the docs files after every task and before every commit; commit + push at the end of each phase on branch `claude/quirky-tesla-kvnvgk`.
- Code is commented for presentation use (the group must explain it).

## Pointers
- `docs/GUIDE.md` — one-page guide for the group (done so far, cloud vs local, checklist, evaluation)
- `docs/PLAN.md` — phases, checklists, definition of done
- `docs/PROGRESS.md` — running log + next steps
- `docs/DECISIONS.md` — decisions with reasons/alternatives, user answers
- `docs/KAGGLE_RUNS.md` — every Kaggle run

## Workflow cheat-sheet
- Local Spider copy for tests: `kaggle datasets download jeromeblanchet/yale-universitys-spider-10-nlp-dataset --unzip` → `SPIDER_DIR=<dir>/spider`.
- Change src → `pytest` → commit → `bash scripts/upload_src.sh "msg"` → `python notebooks/build_notebooks.py [--smoke NN]` → `kaggle kernels push -p notebooks/<nb> --accelerator NvidiaTeslaT4` → poll `kaggle kernels status` → `kaggle kernels output ... -p <dir>`.
- Demo/app change → test locally → commit → PR merged into main → `bash scripts/deploy.sh` (checks + Kaggle test run). Kaggle notebook 04 always runs GitHub main.
- Kaggle owner slug: `mibrahimsalman022` (no dot).

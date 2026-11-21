# SQLBuddy — project brief

**Start of every session:** read this file, then `docs/PROGRESS.md` (newest entry on top, with "Next steps"), then the relevant phase in `docs/PLAN.md`.

## Goal
University NLP course project: fine-tune a small LLM to translate English questions into SQL and **prove with a held-out test set** whether fine-tuning beats the base model. Grading focuses on NLP methodology and results; the demo is secondary.

## Tech choices (details + reasons in `docs/DECISIONS.md`)
- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Method: LoRA (PEFT) + TRL `SFTTrainer`, fp16 on a Kaggle T4 (no bf16 on T4). Loss on SQL completion only.
- Data: Spider 1.0. Train = `train_spider.json` (7,000) minus a small validation split; **test = dev (1,034), never trained on**.
- Spider SQLite DBs: Kaggle dataset `jeromeblanchet/yale-universitys-spider-10-nlp-dataset` (original release, contains `spider/database/`), attached as kernel input.
- Prompt: schema as CREATE TABLE statements built from the SQLite file + question → SQL only. Identical prompt for base and fine-tuned.
- Decoding: greedy, `max_new_tokens=256`, batched, then SQL extraction post-processing.
- Metrics: execution accuracy (primary), exact match (Spider official exact-set-match), valid-SQL rate; all by Spider hardness (easy/medium/hard/extra) via vendored official `evaluation.py`.
- GPU work runs only on Kaggle (pushed with the `kaggle` CLI). Adapter goes to HF Hub `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora` (private).

## Repo structure
```
notebooks/   Kaggle notebooks, one folder each with kernel-metadata.json
src/         shared code: schema/prompt building, SQL extraction, evaluation (tested on CPU here)
app/         Gradio demo
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
- `docs/PLAN.md` — phases, checklists, definition of done
- `docs/PROGRESS.md` — running log + next steps
- `docs/DECISIONS.md` — decisions with reasons/alternatives, user answers
- `docs/KAGGLE_RUNS.md` — every Kaggle run

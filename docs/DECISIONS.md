# Decisions

Format: decision — reason — alternatives considered. User answers are marked **(user)**.

## Given by the user (project brief)
- **(user)** Base model `Qwen/Qwen2.5-Coder-1.5B-Instruct` — ungated, code-pretrained, fits a T4 with plain fp16 LoRA.
- **(user)** LoRA via PEFT + TRL SFTTrainer; fp16 (T4 has no bf16); QLoRA only if memory requires.
- **(user)** Dataset Spider; dev (1,034) is the held-out test set; validation carved out of train; never train on dev.
- **(user)** Prompt = CREATE TABLE schema + question, target = SQL only; same prompt for base and fine-tuned.
- **(user)** Greedy decoding, max_new_tokens ≈ 256, batched, SQL extraction post-processing.
- **(user)** LoRA hyperparameters: r=16, alpha=32, dropout 0.05, attention + MLP projections, 2 epochs, lr 2e-4, cosine, max seq len ~1024.

## Phase 0
- **Spider DB source = Kaggle dataset `jeromeblanchet/yale-universitys-spider-10-nlp-dataset`.** It is the original Spider 1.0 release (README, `train_spider.json`, `dev.json`, `tables.json`, `database/` with 166 DBs). Verified: all 160 DBs used by train+dev exist; 7,000/1,034 counts match HF `xlangai/spider`; most-downloaded mirror (5k+). Alternatives: original Google Drive zip (works but needs manual upload to Kaggle), other smaller mirrors (less used).
- **Spider hardness = vendored official `evaluation.py`/`process_sql.py`** from `taoyds/spider` (Apache-2.0) instead of a re-implementation — guarantees the exact official easy/medium/hard/extra labels. The mirror's JSON already contains the parsed `sql` field that `eval_hardness` needs.
- **Shared code delivered to Kaggle as a private dataset `sqlbuddy-src`** (versioned) — notebooks import the same CPU-tested `src/` instead of copy-pasted code. Alternative: embed code into notebooks with `%%writefile` (self-contained but duplicates code); git clone (repo may be private).
- **HF upload happens from this environment, not from Kaggle** — no HF token has to be stored in Kaggle secrets.
- **Kaggle CLI 2.2.5** works with the `KGAT_` token from `KAGGLE_API_TOKEN` directly; no legacy key needed.

## User answers after Phase 0 (2026-11-21)
- **(user) Validation split = hold out whole databases** (~5% of train examples, from DBs not used for training). Reason: Spider dev uses databases unseen in training, so a DB-disjoint validation set makes val loss a fair proxy for cross-database generalization. Alternative: random 5% of examples (optimistic, DBs overlap).
- **(user) Exact match = official Spider exact-set-match** (component-level, values ignored). Reason: standard metric, comparable to the literature. Alternatives: normalized string match; both.
- **(user) Training data = `train_spider.json` only (7,000)**, not `train_others.json`. Reason: matches the brief and HF copy; train_others has a different style and +24% training time.
- **(user) Phase 1 plan approved** as written in PLAN.md.

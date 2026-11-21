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

## Phase 1 (implementation decisions)
- **Questions + gold SQL from HF `xlangai/spider`; only DB files + `tables.json` from the Kaggle mirror.** Train is identical in both, but HF dev is the *newer corrected* dev set (44 queries differ from the 2020 zip). Using the HF dev gold SQL reproduces the official difficulty counts exactly: easy 248 / medium 446 / hard 174 / extra 166.
- **Hardness = official `eval_hardness` on the gold SQL re-parsed with official `get_sql`** (as in official `evaluate()`), not the stored `sql` field of the old JSON (which gives 252/469/153/160).
- **Schema text = canonical CREATE TABLE rebuilt from `PRAGMA table_info` / `foreign_key_list`** (column, type, PK, FKs), not the raw `sqlite_master` DDL. Reason: Spider's raw DDL is very inconsistent (comments, quoting, CHECKs); canonical DDL is uniform, valid SQLite (tested for all 166 DBs), and works the same for any uploaded DB in the demo. Identifiers are quoted when they contain special characters or are SQL keywords (a Spider column is literally named `From`). Alternative: raw DDL; adding sample rows (more tokens, left out to keep the prompt as in the brief).
- **Prompt:** system message ("expert SQLite assistant … return only the SQL query") + user message `Database schema:\n<DDL>\n\nQuestion: <q>`, through the Qwen chat template. Training target = gold SQL with whitespace normalized (Spider gold has `a ,  b` spacing).
- **Execution-accuracy comparison rules:** rows as multisets (order ignored) unless gold has ORDER BY; column permutations allowed (≤6 columns); floats rounded to 6 decimals; 30 s timeout per query; prediction error/timeout = wrong. Close to the test-suite evaluation's approach. Known limitation: accidental matches (e.g. `count(*)` of two different tables that both have 6 rows — found in our own tests).
- **Exact match:** official `eval_exact_match` after the official value/column rebuild steps (values ignored, FK columns unified via `tables.json`). Unparseable prediction → 0.
- **Token lengths (Qwen tokenizer, full prompt + SQL):** dev mean 354, max 898 → no truncation needed at test time. Train mean 457, p95 1005, max 2328; 311 examples > 1024, 88 > 1536. → handled in the Phase 2 plan.
- **Validation split (DB-disjoint, seed 42, 5%):** 352 examples from 8 DBs (body_builder, climbing, gas_company, local_govt_mdm, music_1, protein_institute, tracking_grants_for_research, workshop_paper); training keeps 6,648.
- **Library pins:** latest releases at least ~3 weeks old (the newest were days old): transformers 5.22.1, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, trl 1.17.1, huggingface_hub 2.4.0, nltk 3.9.2, gradio 6.31.0, torch 2.14.1 (local; Kaggle uses its preinstalled torch, version logged in run_info.json).
- **Kaggle owner slug is `mibrahimsalman022`** (the `KAGGLE_USERNAME` value `m.ibrahimsalman022` contains a dot that Kaggle drops in URLs/ids). Scripts derive the slug by removing dots.
- **Notebooks are generated by `notebooks/build_notebooks.py`** (cells as Python source) so they are reviewable and reproducible.
- **Kaggle accelerator:** `--accelerator NvidiaTeslaT4` (T4 ×2 machine; we use one GPU).

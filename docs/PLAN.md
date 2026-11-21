# SQLBuddy — Plan

Status legend: `- [ ]` todo, `- [x]` done. Each phase ends with: update docs → commit → push → short status update to the user. Each phase's detailed plan must be approved by the user before implementation.

---

## Phase 0 — Environment check & scaffolding ✅ (awaiting approval of the plan below)
- [x] Install Kaggle CLI (2.2.5) and verify auth with `KAGGLE_API_TOKEN` (`kaggle kernels list --mine` works)
- [x] Verify network: huggingface.co, Kaggle API, raw.githubusercontent.com (official Spider scripts) reachable
- [x] Verify HF token (fine-grained, user `ibrahimsalman022`)
- [x] Find Spider SQLite source: Kaggle `jeromeblanchet/yale-universitys-spider-10-nlp-dataset` (166 DBs, all 160 needed by train+dev present; JSON matches HF `xlangai/spider`)
- [x] Repo structure, `.gitignore`, `CLAUDE.md`, docs files
- [ ] User approves plan / answers open questions (see `docs/DECISIONS.md` → "Open questions")

**Done when:** CLI auth works, sources confirmed, docs + structure pushed, user approved plan.

---

## Phase 1 — Data prep + base model evaluation

### Tasks
- [ ] Vendor official Spider `evaluation.py` + `process_sql.py` (Apache-2.0) into `src/spider_eval/` with license header; minimal py3.13 compatibility fixes only.
- [ ] `src/schema.py`: read a SQLite file → CREATE TABLE text. Use the `sqlite_master.sql` DDL, normalized (strip comments/extra whitespace) so the model sees real types, PKs and FKs. Fallback: build DDL from `PRAGMA table_info` / `foreign_key_list`. Cache per db_id.
- [ ] `src/prompts.py`: one function `build_messages(schema, question)` → chat messages (system: "You are a SQLite expert…return only SQL"; user: schema + question). Uses the tokenizer chat template. Same function for base, fine-tuned, demo.
- [ ] `src/sql_extract.py`: extract SQL from model output (```sql fences, leading "SQL:", trailing prose, keep first statement, strip `;`).
- [ ] `src/execution.py`: run SQL on SQLite read-only (`file:...?mode=ro`, `uri=True`) with a timeout (progress handler, 30 s); return rows or error.
- [ ] `src/evaluate.py`: for each example compute
  - `exec_match`: predicted vs gold result sets; multiset (order-insensitive) comparison unless gold has `ORDER BY` (then order-sensitive). Invalid/timeout = wrong.
  - `exact_match`: official Spider exact-set-match (component matching, values ignored). If the official parser can't parse a prediction → 0.
  - `valid_sql`: prediction executes without error.
  - `hardness`: official `Evaluator.eval_hardness` on the gold parsed SQL.
  - Aggregate: overall + per hardness → `metrics.csv`; per-example → `predictions.csv`.
- [ ] `src/data.py`: load Spider (HF `xlangai/spider` for questions/SQL; DB files + parsed `sql` from the Kaggle mirror), make the train/validation split (fixed seed), check dev never overlaps.
- [ ] CPU tests (`tests/`, pytest): schema builder on 3 DBs, prompt shape, extraction on ~15 messy outputs, evaluator with gold-as-prediction must give 100% EX/EM on all 1,034 dev examples (sanity check of the whole pipeline), and some known-wrong predictions give 0.
- [ ] Upload `src/` as private Kaggle dataset `sqlbuddy-src` (versioned with `kaggle datasets version`) so notebooks import the same tested code.
- [ ] `notebooks/01_baseline_eval/` (`.ipynb` + `kernel-metadata.json`, GPU T4, internet on, inputs: Spider mirror + `sqlbuddy-src`). Param `N_EXAMPLES`.
  - [ ] Run 1: `N_EXAMPLES=50` → measure time, check outputs look sane, print library versions → pin them.
  - [ ] Run 2: full dev (1,034).
- [ ] Download outputs into `results/baseline/` (predictions.csv, metrics.csv, run_info.json).

### Testing
CPU unit tests + gold-as-prediction sanity check before any push; 50-example Kaggle smoke run before the full run.

### Expected Kaggle GPU time
Smoke run ~10 min (incl. model download). Full dev with HF `generate`, batch 16, greedy, 256 tokens: ~25–45 min. Total ≈ 1 h of quota.

### Risks
- Long schemas (some Spider DBs have many tables) → prompt length. Mitigation: measure token lengths; sort by length for batching; cap at 2,048 prompt tokens (report how many hit the cap — expect none/few).
- Some gold queries are slow on SQLite → timeout per query.
- Official eval script is py2-era code; may need small fixes.

---

## Phase 2 — LoRA fine-tuning

### Tasks
- [ ] `notebooks/02_train_lora/`: build training examples with exactly the Phase-1 prompt (chat template) + gold SQL as assistant turn; mask prompt tokens (loss only on the SQL completion, via TRL `assistant_only_loss`/completion-only, or a custom collator if needed).
- [ ] LoRA: r=16, alpha=32, dropout=0.05, targets `q,k,v,o,gate,up,down_proj`; 2 epochs; lr 2e-4; cosine; warmup 3%; max seq len 1024 (drop/truncate & report the few longer examples); fp16; gradient checkpointing; effective batch 16 (e.g. 4 × grad-accum 4).
- [ ] Eval on validation split every ~200 steps; log train/val loss; save `loss_history.csv` + `loss_curve.png`.
- [ ] Checkpoint every ~200 steps (keep last 2) so a timeout doesn't lose the run.
- [ ] Short smoke run (~30 steps) first to check memory + step time → report expected full run time to user before launching the full run.
- [ ] Download adapter (outside git), upload to HF `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora` (private) from here with HF_TOKEN.

### Expected Kaggle GPU time
~6,650 examples × 2 epochs ≈ 830 optimizer steps; estimate 1.5–3 h on a T4 (to be measured in the smoke run). Kaggle limit is 12 h/session.

### Risks
fp16 overflow / NaN loss (mitigation: LoRA params in fp32, grad clipping 1.0, fp16 AMP); OOM on long schemas (reduce batch, raise grad-accum).

---

## Phase 3 — Fine-tuned evaluation
- [ ] `notebooks/03_finetuned_eval/`: same code path as 01 (shared `src/`), adapter loaded from the Phase-2 kernel output (or HF), same prompts/decoding/metrics.
- [ ] `results/finetuned/` predictions + metrics; `results/comparison.csv` (base vs FT, overall + per hardness, EX/EM/valid, deltas).
- Expected GPU time ~30–45 min.

---

## Phase 4 — Analysis (most important for grading)
- [ ] Charts (PNG): EX base vs FT grouped bars (overall + per hardness), valid-SQL rate, training/val loss curve.
- [ ] `results/error_analysis.md`: categorize ~50 FT failures (wrong table, wrong column, JOIN, GROUP BY/aggregation, condition/value, nesting, invalid SQL) — rule-assisted pre-labels then manual review; 5 fixed-by-FT examples, 5 still-wrong examples (question, gold, base pred, FT pred); regressions (correct→wrong) discussed.
- [ ] Statistics: flip counts (wrong→correct, correct→wrong), McNemar's exact test (scipy), 95% bootstrap CI for EX difference.
- CPU only, no GPU.

---

## Phase 5 — Demo
- [ ] `sample_db/make_university_db.py` → `sample_db/university.db` (students, teachers, courses, enrollments, grades; ~50–200 rows, fixed seed).
- [ ] `app/app.py` (Gradio): pick university.db or upload .sqlite; schema view; question box; Base / Fine-tuned toggle; generated SQL + result table. Read-only SQLite (`mode=ro`) and SELECT-only guard (single statement, must start with SELECT/WITH; reject anything else).
- [ ] CPU fp32 loading from HF (base + adapter via PEFT); cache both models.
- [ ] `notebooks/04_demo/`: same app on GPU with `share=True` link (backup).
- [ ] Test locally on CPU here (load + 2 queries).

---

## Phase 6 — Documentation
- [ ] README.md: problem, why text-to-SQL, dataset, method + mermaid pipeline, training setup, results table + charts, error analysis summary, limitations, reproduction (Kaggle steps), demo on Windows/Mac.
- [ ] `PRESENTATION_NOTES.md`: 10-slide outline with key numbers/charts.
- [ ] Final summary to user: results table, file list, laptop demo commands.

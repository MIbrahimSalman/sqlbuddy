# SQLBuddy — Plan

Status legend: `- [ ]` todo, `- [x]` done. Each phase ends with: update docs → commit → push → short status update to the user. Each phase's detailed plan must be approved by the user before implementation.

---

## Phase 0 — Environment check & scaffolding ✅
- [x] Install Kaggle CLI (2.2.5) and verify auth with `KAGGLE_API_TOKEN` (`kaggle kernels list --mine` works)
- [x] Verify network: huggingface.co, Kaggle API, raw.githubusercontent.com (official Spider scripts) reachable
- [x] Verify HF token (fine-grained, user `ibrahimsalman022`)
- [x] Find Spider SQLite source: Kaggle `jeromeblanchet/yale-universitys-spider-10-nlp-dataset` (166 DBs, all 160 needed by train+dev present; JSON matches HF `xlangai/spider`)
- [x] Repo structure, `.gitignore`, `CLAUDE.md`, docs files
- [x] User approved plan + answered open questions (see DECISIONS.md)

**Done when:** CLI auth works, sources confirmed, docs + structure pushed, user approved plan. ✅

---

## Phase 1 — Data prep + base model evaluation ✅

### Tasks
- [x] Vendor official Spider `evaluation.py` + `process_sql.py` (Apache-2.0) into `src/spider_eval/` with license header; minimal py3.13 compatibility fixes only.
- [x] `src/schema.py`: SQLite file → canonical CREATE TABLE text rebuilt from `PRAGMA table_info` / `foreign_key_list` (types, PKs, FKs); changed from raw `sqlite_master` DDL — see DECISIONS.md. Cached per file.
- [x] `src/prompts.py`: one function `build_messages(schema, question)` → chat messages (system: "You are a SQLite expert…return only SQL"; user: schema + question). Uses the tokenizer chat template. Same function for base, fine-tuned, demo.
- [x] `src/sql_extract.py`: extract SQL from model output (```sql fences, leading "SQL:", trailing prose, keep first statement, strip `;`).
- [x] `src/execution.py`: run SQL on SQLite read-only (`file:...?mode=ro`, `uri=True`) with a timeout (progress handler, 30 s); return rows or error.
- [x] `src/evaluate.py`: for each example compute
  - `exec_match`: predicted vs gold result sets; multiset (order-insensitive) comparison unless gold has `ORDER BY` (then order-sensitive). Invalid/timeout = wrong.
  - `exact_match`: official Spider exact-set-match (component matching, values ignored). If the official parser can't parse a prediction → 0.
  - `valid_sql`: prediction executes without error.
  - `hardness`: official `Evaluator.eval_hardness` on the gold parsed SQL.
  - Aggregate: overall + per hardness → `metrics.csv`; per-example → `predictions.csv`.
- [x] `src/data.py`: load Spider (HF `xlangai/spider` for questions/SQL; DB files + `tables.json` from the Kaggle mirror), make the train/validation split (fixed seed), check dev never overlaps.
- [x] CPU tests (`tests/`, pytest): schema builder on 3 DBs, prompt shape, extraction on ~15 messy outputs, evaluator with gold-as-prediction must give 100% EX/EM on all 1,034 dev examples (sanity check of the whole pipeline), and some known-wrong predictions give 0.
- [x] Upload `src/` as private Kaggle dataset `sqlbuddy-src` (versioned with `kaggle datasets version`) so notebooks import the same tested code.
- [x] `src/generation.py` (model loading, batched greedy generation) + `src/run_eval.py` (end-to-end eval CLI shared by notebooks 01 and 03); CPU run on 4 dev examples OK.
- [x] `notebooks/build_notebooks.py` generates the notebooks.
- [x] `notebooks/01_baseline_eval/` (`.ipynb` + `kernel-metadata.json`, GPU T4, internet on, inputs: Spider mirror + `sqlbuddy-src`). Param `N_EXAMPLES`.
  - [x] Run 1: `N_EXAMPLES=50` → measure time, check outputs look sane, print library versions → pin them.
  - [x] Run 2: full dev (1,034). EX 56.96 / EM 32.40 / valid 76.60; 6 min total.
- [x] Download outputs into `results/baseline/` (predictions.csv, metrics.csv, run_info.json).

### Testing
CPU unit tests + gold-as-prediction sanity check before any push; 50-example Kaggle smoke run before the full run.

### Expected Kaggle GPU time
Estimated ~1 h. **Actual: 8 min total** (smoke 2 min, full 6 min; generation 0.22 s/example).

### Risks
- Long schemas (some Spider DBs have many tables) → prompt length. Mitigation: measure token lengths; sort by length for batching; cap at 2,048 prompt tokens (report how many hit the cap — expect none/few).
- Some gold queries are slow on SQLite → timeout per query.
- Official eval script is py2-era code; may need small fixes.

---

## Phase 2 — LoRA fine-tuning ✅ (max_length 1536)

### Data
- Train 6,648 / validation 352 (DB-disjoint, 8 held-out DBs, seed 42) from `train_spider.json`; dev is never touched.
- Each example = TRL *prompt–completion* format: `prompt` = `build_messages(schema, question)` (system + user, identical to eval), `completion` = assistant turn with whitespace-normalized gold SQL.
- `completion_only_loss=True`: loss only on the SQL tokens + `<|im_end|>`. **Verified on CPU** with TRL 1.17.1 (decoded labels = `SELECT count(*) FROM t<|im_end|>`), and the prompt tokens are byte-identical to the eval prompt.
- Examples longer than `max_length` are **dropped, not truncated** (truncation would cut the SQL target). Measured: 311 train examples > 1024 tokens, 88 > 1536.

### Files
- `src/train.py`: builds datasets, LoRA config, `SFTConfig`, a loss-logging callback; saves adapter + `loss_history.csv` + `loss_curve.png` + `train_info.json`. Same "shared tested code" approach as eval.
- `notebooks/02_train_lora/` (generated by `build_notebooks.py`), param `MAX_STEPS` (smoke) / full.
- `scripts/upload_adapter.py`: download adapter from kernel output → HF `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora` (private) with model card.

### Hyperparameters
- LoRA r=16, alpha=32, dropout 0.05, targets q/k/v/o/gate/up/down_proj (≈18.5M trainable params, ~1.2%).
- 2 epochs, lr 2e-4, cosine, warmup 3%, weight decay 0, grad-clip 1.0, seed 42.
- Batch 4 × grad-accum 4 = 16 effective → ~415 steps/epoch, ~830 steps total.
- fp16 AMP; base weights frozen in fp16, LoRA weights fp32 (PEFT default); gradient checkpointing; SDPA attention.
- Single T4 (`CUDA_VISIBLE_DEVICES=0`; the second T4 of the Kaggle machine is unused — avoids DataParallel issues with LoRA + gradient checkpointing).
- Eval (val loss) + checkpoint every 100 steps, keep last 2; train loss logged every 10 steps.

### Testing
1. CPU: `src/train.py` 2 steps with a tiny Qwen2 test model (masking, saving, loss CSV/plot).
2. Kaggle smoke run: `MAX_STEPS=30` → s/step, peak GPU memory, loss decreasing, no NaN → **report expected full run time to the user before the full run.**
3. Full run; download adapter; reload it locally on CPU and generate 2 queries as a check.

### Expected Kaggle GPU time
**Measured in smoke run: 10.96 s/step, 5.4 GB peak → full run ≈ 820 steps ≈ 2.5 h (under the 3 h auto-go threshold).** Original estimate: smoke ~10 min. Full: ~1–2 h (≈10k tokens/step × 830 steps on a T4 with gradient checkpointing, + 8 val evals). Measured in the smoke run.

### Risks
fp16 overflow/NaN (fp32 LoRA weights, grad clipping, AMP loss scaling); OOM on long examples (reduce per-device batch to 2, grad-accum 8); Kaggle 12 h limit (far away; checkpoints anyway).

## Phase 3 — Fine-tuned evaluation ✅
- [x] `notebooks/03_finetuned_eval/` generated: same `src/run_eval.py` as notebook 01; only change = `--adapter` (02 kernel output attached as `kernel_sources`, merged into the base with `merge_and_unload`). Same prompts, greedy, 256 tokens, batch 16.
- [ ] Smoke run on 50 examples (checks adapter loading + output format), then the full dev set.
- [ ] Download to `results/finetuned/` (predictions.csv, metrics.csv, run_info.json, log).
- [ ] `src/compare.py` → `results/comparison.csv`: rows = level (easy/medium/hard/extra/all) × columns base/FT/delta for exec_acc, exact_match, valid_sql (+ counts).
- Testing: compare.py unit-tested on CPU (base vs base must give delta 0). Expected GPU time ~10 min total (baseline took 6 min; merged LoRA has the same speed).
- Risk: adapter path inside the kernel input differs → notebook globs for `adapter_config.json` outside `checkpoint-*`.

## Phase 4 — Analysis ✅ (CPU only)
- [ ] `src/analysis.py` (run locally, outputs in `results/`):
  - `exec_acc_by_level.png` grouped bars base vs FT (overall + 4 levels, value labels, n per level);
  - `valid_sql_rate.png` (base vs FT, overall + per level); `exact_match_by_level.png`;
  - `loss_curve.png` (train + validation loss, from `loss_history.csv`);
  - `flips.csv` + statistics: wrong→correct, correct→wrong, both right/wrong; **McNemar's exact test** (binomial on discordant pairs, scipy) and a **95% paired bootstrap CI** (10,000 resamples) for the EX difference; also per level.
- [ ] Error categories for FT failures: automatic pre-label by comparing gold vs predicted SQL structure (parsed with the official Spider parser / `sqlglot`): invalid SQL; wrong table; wrong column; missing/incorrect JOIN; GROUP BY / aggregation; nested query / set operation; wrong condition / value; ORDER BY / LIMIT; other. Then I review ~50 sampled failures (stratified by level, seed 42) by hand and correct labels → `results/error_categories.csv`.
- [ ] `results/error_analysis.md`: category table + discussion; 5 fixed-by-FT examples and 5 still-wrong (question, gold, base pred, FT pred, why); regressions (correct→wrong) with examples; error-type shift base vs FT (e.g. fewer "no such column"); caveats (accidental EX matches, EM vs EX disagreements).
- Done when: all charts + error_analysis.md exist and numbers match comparison.csv.

## Phase 5 — Demo ✅
- [ ] `sample_db/make_university_db.py` (fixed seed, stdlib only) → `sample_db/university.db`: tables `students`, `teachers`, `courses` (FK teacher), `enrollments` (student, course, semester), `grades` (enrollment, score, letter); ~150 rows in total; realistic names/departments.
- [ ] `app/sqlbuddy_core.py`: model manager (lazy-load base; FT = base + adapter from HF `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora` via PEFT; keeps both loaded, using `disable_adapter()` for base → only one copy of the weights in RAM, ~6 GB fp32); `safe_select()`: single statement, must start with SELECT/WITH, rejects ; chains / PRAGMA / ATTACH etc., executes via `src.execution` read-only with timeout and row cap (200).
- [ ] `app/app.py` (Gradio 6.31): DB choice (sample / upload .sqlite/.db), schema view, question box + example questions, Base / Fine-tuned radio, outputs: generated SQL, result table, timing, error message. Reuses `src/schema.py` + `src/prompts.py` + `src/sql_extract.py` → identical prompt to evaluation.
- [ ] HF access: the adapter repo is private → the laptop needs `HF_TOKEN` (read) or `huggingface-cli login`; documented. (Alternative if you prefer: make the adapter repo public later.)
- [ ] `notebooks/04_demo/`: same app on GPU, `share=True` public link, adapter taken from the 02 kernel output (no token needed); runs until stopped / session timeout.
- [ ] Tests: unit tests for `safe_select`; CPU smoke test here (load model, 2 questions on university.db, base and FT).

## Phase 6 — Documentation ✅
- [x] README.md: problem, why text-to-SQL, dataset, method + mermaid pipeline, training setup, results table + charts, error analysis summary, limitations, reproduction (Kaggle steps), demo on Windows/Mac.
- [x] `PRESENTATION_NOTES.md`: 10-slide outline with key numbers/charts.
- [x] Final summary to user: results table, file list, laptop demo commands.

## Phase 7 — Demo polish, latency, extra evaluation, slide deck (self-approved, user request 2026-11-22)
- [x] UI redesign: three steps (database → question → results), upload validation, schema preview, examples, Fine-tuned default, side-by-side Compare panels, explicit states (loading / queued / generating / running SQL / done / error / cancelled), device + mode + measured time, Stop, no duplicate submissions, plain-language SQL errors, read-only protections unchanged. Responsive at phone width.
- [x] Latency investigation (`docs/PERFORMANCE.md`): profile load / prefill / decode; output-preserving speed-ups only (background load + warm-up, prompt-prefix KV cache, streaming, MPS on Macs).
- [x] UI tests (`tests/test_app_ui.py`); notebook 04 re-verified on a T4.
- [x] Per-database breakdown of the dev results (`results/per_database.csv`, `exec_acc_by_database.png`).
- [x] Out-of-domain test: 30 hand-written questions on `university.db`, gold SQL fixed before running the models (`sample_db/university_test.json`, `src/ood_eval.py`, `results/ood/`).
- [x] Slide deck (claude.ai Slides artifact, exportable to PowerPoint/PDF) + updated PRESENTATION_NOTES / README.

## Phase 8 — Hosting the demo + one-command updates (self-approved, user request 2026-11-22)
- [x] Adapter repo on Hugging Face made **public** (user OK) → no login/token needed anywhere.
- [x] Hugging Face Space: attempted, **refused** (Gradio Spaces on free CPU now need a PRO subscription). User chose Kaggle only (free GPU).
- [x] Notebook 04 downloads the app from **GitHub main** at the start of every run (prints the commit), adapter from the public HF repo → every merged design change reaches Kaggle automatically. No Kaggle inputs needed anymore.
- [x] Run length: editor "Run all" (Interactive) = 90 min; CLI push (Batch) = self-test + 1 min.
- [x] `scripts/deploy.sh`: tests → check commit is in GitHub main → refresh `sqlbuddy-src` → Kaggle test run of notebook 04, printing the commit Kaggle ran + both models' answers.
- [x] Docs: README §7, GUIDE, PROGRESS, DECISIONS, KAGGLE_RUNS, CLAUDE.md.

# Progress log (newest first)

## 2026-11-22 — Slide deck finalised

**Done**
- Cover: Fatima Malik (23L-0826), Mohammad Ibrahim Salman (23L-0723), CS4063 – Natural Language Processing. The date was removed, as the user asked.
- Demo slide (13) now describes the Kaggle GPU demo (~1–4 s per question, public link, latest app from GitHub), with the laptop CPU as a fallback. The speaker notes have the Kaggle steps.
- PRESENTATION_NOTES.md and GUIDE.md updated to match. Docs-only change, so deploy.sh was not run.

**Next steps**
- Share the deck from its Share menu (and optionally export it as PowerPoint/PDF for an offline copy).
- Presentation day: start notebook 04 from the Kaggle editor about 5 minutes early.

## 2026-11-22 — Standing rule for Claude: run deploy.sh after code changes

**Done**
- At the user's request, added a "RULE: after every code change, run `scripts/deploy.sh`" section to `CLAUDE.md`, which Claude reads at the start of every session. After any change to `app/`, `src/`, `sample_db/`, the notebook builder or the requirements files, Claude must test, merge into `main`, run `bash scripts/deploy.sh` (it must end with `RESULT: OK`), log the Kaggle run and report the result. Changes to docs or results files alone skip it.
- No new files, as requested. This change is docs-only, so deploy.sh was not run.

**Next steps**
- Unchanged: fill in the deck's cover placeholders and share the deck; on presentation day run notebook 04 from the Kaggle editor.

## 2026-11-22 — Phase 8: demo hosting on Kaggle (from GitHub main), public adapter, deploy.sh

**Done**
- Adapter repo on Hugging Face made **public** (user OK). Laptops and Kaggle load it with no login; the app's wording is updated.
- **Hugging Face Space attempted and dropped.** Creating it returned 402: Gradio Spaces on free CPU now need PRO. User chose "Kaggle only, free". The Space files were removed.
- **Notebook 04 now downloads the app from GitHub main at the start of every run** (prints the commit). The adapter comes from the public HF repo and the notebook has no Kaggle inputs. Editor "Run all" = 90 min demo; CLI push = self-test + 1 min.
- **`scripts/deploy.sh`** (one command after app changes): tests → commit must be in GitHub main → refresh `sqlbuddy-src` → Kaggle test run of notebook 04 → prints the commit Kaggle ran and checks that the adapter loaded and the fine-tuned answer succeeded. It refuses with uncommitted or unmerged work, or while notebook 04 is running.
- Fixed a deploy.sh bug found in testing: right after a push, `kaggle kernels status` can still report the previous run → the script now polls `<kernel>/<version>` and explains cancelled runs.
- **End-to-end test (run 14): OK.** Kaggle ran GitHub main `bc0cbd5` on a Tesla T4 (fp16) with the adapter loaded; run type Batch → 1 min; public link created.
- Docs: README §7 (Kaggle demo steps, deploy.sh, no HF login), GUIDE, PLAN (Phase 8), DECISIONS, KAGGLE_RUNS (runs 11–14), CLAUDE.md.

**Next steps**
- Presentation day: Kaggle → `sqlbuddy-04-demo` → Edit → GPU T4 + Internet on → Run all → share the `PUBLIC LINK` (90 min).
- After any app change: merge into main, then `bash scripts/deploy.sh`.
- Still open from Phase 7: fill in the deck's cover placeholders and share the deck.

## 2026-11-22 — Group guide

**Done**
- User asked for a summary: what has been done, how to run locally, what ran in the cloud, what to check and how to evaluate. Wrote `docs/GUIDE.md` (one page) and linked it from the README and CLAUDE.md.

**Next steps**
- Unchanged: fill in the deck's cover placeholders and share the deck; give group members HF read access; start the app early on presentation day.

## 2026-11-22 — Phase 7: demo redesign, latency investigation, extra evaluation, slide deck

**User feedback:** everything works; improve the interface, explain why queries are slow, continue with evaluation and presentation.

**Done**
- **UI** (`app/app.py`, `app/sqlbuddy_core.py`):
  - three steps; DB card with table/row counts; upload validation; schema preview;
  - Fine-tuned as the default; Compare panels side by side (stated as two generations);
  - explicit states (loading / queued / generating / running SQL / done / SQL error / cancelled);
  - device, mode, tokens and measured time; streamed SQL; Stop; Generate disabled while running; plain-language SQL errors;
  - responsive at phone width (fixed a non-wrapping schema box);
  - read-only protections unchanged.
  - Tested with Playwright (compare, stop, upload, phone) and on a Kaggle T4 (04 v2).
- **Latency** (`docs/PERFORMANCE.md`):
  - Hardware-bound: ~0.16 s per token on 4 cores, scaling with core count; generation always stops at the end of the SQL.
  - Added output-preserving speed-ups: background load + warm-up, prompt-prefix KV cache (first token 2.3 s → 0.6 s, identical output 8/8), streaming, MPS on Macs.
  - Rejected: prompt-lookup decoding (50 % slower on CPU), int8 and smaller models (would change the evaluated model).
- **Evaluation:**
  - Per-database breakdown: 18 of 20 unseen DBs improve (`results/per_database.csv`, `exec_acc_by_database.png`).
  - Out-of-domain test on our own DB (30 pre-written questions): base 36.7 % → fine-tuned 73.3 %, 11 fixed / 0 broken, McNemar p = 0.001 (`results/ood/`).
- **Presentation:** 15-slide deck with speaker notes at https://claude.ai/artifact/GVSZ3uR7kogY9rAMugzxJH (private until shared; exports to PowerPoint/PDF). `PRESENTATION_NOTES.md` rewritten to match, with a demo script and Q&A.
- Tests: 56 pass.

**Next steps**
- Fill in the cover placeholders (group members, course, date) and share the deck with the group.
- Give group members read access to the HF adapter repo.
- On presentation day: start the app early, or run `04_demo` (90 min) as the GPU backup.

## 2026-11-22 — Phases 5 + 6: demo verified, README + presentation notes

**Done**
- Demo tested end-to-end on CPU with the private HF adapter (`Compare both` on all 6 examples): FT correct on 4/6, base on 2/6; both fail "students per department" (honest schema-linking example). DROP is blocked. UI served via HTTP; screenshot `results/demo_screenshot.png`.
- Demo polish: "(no result)" table instead of a dummy header, floats rounded to 2 decimals, wrapped + height-capped SQL boxes (CSS passed to `launch()`, Gradio 6).
- Notebook 04 verified on a T4 (smoke v1): adapter loads, self-test identical to CPU, public link works.
- Notebook 04: self-test cell (one question through both models), prints `PUBLIC LINK`, `--smoke 04` = 1-minute run.
- `requirements-demo.txt` (minimal laptop install). README (mermaid pipeline, data, method, results, error summary, limitations, Windows/macOS demo, Kaggle reproduction, layout) and `PRESENTATION_NOTES.md` (10 slides + Q&A).

**Next steps**
- Optional: grant group members read access to the HF adapter repo (or make it public) so they can run the fine-tuned demo on their laptops.
- Before presenting: run `04_demo` (full, 90 min) as the GPU backup and rehearse with the demo questions in README §7.

## 2026-11-22 — Phases 3 + 4 complete: fine-tuning beats the base model

**Results (Spider dev, 1,034 held-out questions)**
| | EX | EM | valid SQL |
|---|---|---|---|
| base | 56.96 | 32.40 | 76.60 |
| fine-tuned | **73.11** | **67.99** | **91.78** |
- EX +16.15 pp (95 % paired bootstrap CI 13.2–19.1); 220 fixes vs 53 regressions; exact McNemar p = 2.4e-25; significant at every level (extra: +9.6 pp, p = 0.014; hard: +24.1 pp).

**Done**
- 03 smoke (50) then full run; outputs in `results/finetuned/`. `src.compare` → `comparison.csv`, `flips.csv`, `statistics.json`.
- `src.analysis` → 3 bar charts + loss curve, `error_types.csv`; fixed sample rounding (49 → exactly 50 rows).
- Manual labels for 50 FT failures in `error_sample.csv`; `results/error_analysis.md` (fixes, 5+5 examples, regressions, limitations).
- Key insights: alias errors (`no such column: T2.x`) 201 → 61; FT copies Spider style (0 % INNER JOIN, 0.54 joins/query ≈ gold 0.50) → EM gain (+35.6) partly style; remaining bottleneck = schema linking (27/50) + cell values never seen (5/50); 2/50 gold/eval artifacts; EX false positive example test_197.

**Next steps**
1. Phase 5: demo test with the HF adapter (CPU, base + fine-tuned, Compare mode); optional 04 GPU notebook check.
2. Phase 6: README (mermaid, results, charts, limitations, reproduction, demo commands), `PRESENTATION_NOTES.md`.

## 2026-11-22 — Phase 2 complete: LoRA adapter trained and uploaded

**Done**
- Full run (kernel v4): 2 epochs = 820 steps, 2 h 10 min on a T4 (9.49 s/step, 5.43 GB peak). 18.5 M trainable params (1.18 % of 1.56 B).
- Train loss 0.255 → 0.036. Validation loss (352 examples, 8 unseen DBs): 0.190 (step 100) → **min 0.184 at step 200** → 0.196 at end of epoch 1 → 0.223 final.
- Logs in `results/training/` (loss_history.csv, loss_curve.png, train_info.json). Adapter uploaded privately to HF `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora` (not in git).
- CPU check on `university.db`: fine-tuned model fixes the base model's wrong-alias error on "average score per course" (`T2.title` → `T1.title`).

**Observation (for the report)**
- Validation *loss* rises in epoch 2 while train loss keeps falling: the model becomes over-confident on token level (overfitting in loss terms). We keep the pre-registered choice (final adapter after 2 epochs) and judge it by execution accuracy on the test set, not by loss; only the last 2 checkpoints were kept, so the step-200 checkpoint is not available. Listed as a limitation / future work (early stopping on validation EX).

**Next steps**
1. `03_finetuned_eval` smoke (v1, 50 questions) → full run → `results/finetuned/`.
2. `src.compare`, `src.analysis`, manual error review → `results/error_analysis.md`.
3. Demo test with the HF adapter, Phase 6 docs.

## 2026-11-22 — While training runs: Phase 3/4/5 code prepared

**Done**
- User approved the Phase 3/4/5 plans (run back-to-back); adapter repo stays private (demo uses an HF read token).
- `src/compare.py` (comparison.csv, flips.csv, statistics.json with exact McNemar + paired bootstrap CI) + tests.
- `src/analysis.py` (grouped-bar charts with the validated blue/orange palette, loss curve, automatic error-category pre-labels via sqlglot, stratified 50-failure sample for manual review). Tested on base-vs-base.
- Demo: `sample_db/make_university_db.py` → `university.db` (5 tables, 169 rows); `app/sqlbuddy_core.py` (one copy of base weights, LoRA switched on/off; SELECT-only check + read-only connection + SQLite authorizer + timeout + 200-row cap); `app/app.py` (Gradio: DB choice/upload, schema, Base / Fine-tuned / Compare). CPU test: base answers in ~4 s. 15 safety tests.
- `notebooks/04_demo` (GPU + share link, adapter from the 02 kernel output). `requirements.txt` re-pinned to the verified versions.
- 41 tests pass. Kaggle `sqlbuddy-src` now also ships `app/` + sample DB (layout `src/`, `app/`, `sample_db/`).

**Next steps**
1. When `02_train_lora` v4 completes: download → `results/training/` (loss_history.csv, loss_curve.png, train_info.json); adapter to scratchpad (not git).
2. CPU-check the adapter, upload to HF (`scripts/upload_adapter.py`).
3. Push `03_finetuned_eval` smoke (50) → full; then `python -m src.compare results/baseline results/finetuned results` and `python -m src.analysis results/baseline results/finetuned results/training results`.
4. Manual review of `results/error_sample.csv` → `results/error_analysis.md`.
5. Test the demo with the real adapter (CPU), Phase 6 docs.

## 2026-11-22 — Phase 2 in progress: full LoRA training running on Kaggle

**Done**
- User answers: max_length 1536; auto-start the full run if the smoke estimate is ≤ 3 h; Phase 2 plan approved.
- `src/train.py` (TRL SFTTrainer, prompt–completion data, completion-only loss, LoRA r16/a32, loss logger + curve), CPU-tested with a tiny Qwen2 model.
- Notebook 02 + notebook 03 (fine-tuned eval, adapter taken from the 02 kernel output) + `scripts/upload_adapter.py` (HF upload).
- Smoke run (30 steps): 10.96 s/step, 5.41 GB peak memory, loss 0.221 → 0.203, val loss 0.197. 6,560 train / 352 val (88 dropped > 1536 tokens). Full run launched (~2.5 h, kernel v4).

**Problems & fixes**
- Kaggle image ships torchao 0.10 → PEFT raises ImportError when creating LoRA layers → `%pip uninstall -y torchao` in notebooks 02/03.
- Multi-line cell with bare `pip` lines → SyntaxError (auto-magic only for single-line cells) → explicit `%pip`; all notebook cells are now parse-checked locally with IPython before pushing.
- Note: base model's initial SQL loss is already low (~0.22/token) → consistent with Spider being seen in pre-training.

**Next steps**
1. When kernel v4 finishes: download outputs (adapter outside git; loss_history.csv + loss_curve.png + train_info.json → `results/training/`).
2. Sanity-check the adapter on CPU (load base + adapter, 2 dev questions); upload to HF with `python scripts/upload_adapter.py <adapter> --train-info ... --loss-png ...`.
3. Phase 3: push `03_finetuned_eval` (smoke 50 → full), build `results/comparison.csv`.

## 2026-11-21 — Phase 1 complete: base model evaluation

**Done**
- Vendored official Spider `evaluation.py`/`process_sql.py` (`src/spider_eval/`); wrote `src/` modules: `data`, `schema`, `prompts`, `sql_extract`, `execution`, `evaluate`, `generation`, `run_eval`.
- 23 CPU tests (`tests/test_pipeline.py`) pass, incl. gold-as-prediction = 100% EX/EM/valid on all 1,034 dev examples and the official difficulty counts 248/446/174/166.
- `src/` uploaded as private Kaggle dataset `mibrahimsalman022/sqlbuddy-src` (`scripts/upload_src.sh`); notebooks generated by `notebooks/build_notebooks.py`.
- Kaggle `01_baseline_eval`: smoke run (50 ex, 2 min) then full dev (6 min, generation 224 s).

**Results — base model, Spider dev (1,034)** (`results/baseline/metrics.csv`)

| level | n | exec acc | exact match | valid SQL |
|---|---|---|---|---|
| easy | 248 | 72.18 | 63.31 | 89.11 |
| medium | 446 | 59.19 | 27.13 | 74.89 |
| hard | 174 | 39.66 | 16.09 | 70.69 |
| extra | 166 | 46.39 | 17.47 | 68.67 |
| **all** | 1034 | **56.96** | **32.40** | **76.60** |

Invalid predictions: 225 "no such column" (mostly a column attached to the wrong table alias in JOINs), 15 "no such table". The base model already writes Spider-style SQL (`T1`/`T2` aliases, `a ,  b` spacing) → it has probably seen Spider during pre-training; note as a limitation.

**Problems & fixes**
- HF dev ≠ 2020 zip dev (44 corrected queries) → use HF questions/SQL, mirror only for DB files (exact official difficulty counts).
- Column literally named `From` broke DDL → quote SQL-keyword identifiers.
- `KAGGLE_USERNAME` = `m.ibrahimsalman022` but owner slug is `mibrahimsalman022` → scripts strip dots.
- transformers 5: `apply_chat_template(tokenize=True)` returns a dict → tokenize the text instead.
- Kaggle unpacked the src zip at the dataset root → notebook locates the package by `run_eval.py` + `spider_eval/`.
- Found an accidental execution match in our own tests (two tables with 6 rows each) → documented limitation of EX.

**Next steps**
1. Get user approval for the Phase 2 plan (`docs/PLAN.md`) + answer: max_length 1536 vs 1024; proceed to full training automatically after the smoke run or wait.
2. Write `src/train.py`, CPU test with tiny model, upload src, push `02_train_lora` smoke (MAX_STEPS=30), report expected time, full run, upload adapter to HF.

## 2026-11-21 — Phase 0: environment check & scaffolding

**Done**
- Kaggle CLI 2.2.5 installed; `kaggle kernels list --mine` authenticates with `KAGGLE_API_TOKEN` (account `mibrahimsalman022`).
- HF token valid (user `ibrahimsalman022`, fine-grained). huggingface.co, Kaggle API, raw.githubusercontent.com reachable — no domains need allowing.
- Spider DB source found: Kaggle `jeromeblanchet/yale-universitys-spider-10-nlp-dataset` (original Spider 1.0; 166 DBs; all 160 needed present; counts 7,000/1,034 match HF).
- Official Spider `evaluation.py`/`process_sql.py` downloadable (Apache-2.0) → will be vendored for hardness + exact match.
- Created repo structure, `.gitignore`, `CLAUDE.md`, `docs/*`.

**Problems**: none. (Note: `curl https://www.kaggle.com` returns 404 from this container, but the Kaggle API/CLI works, which is all we need.)

**Next steps**
1. Wait for user's OK on the plan + answers to open questions (validation split, exact-match variant, HF repo name).
2. Start Phase 1: vendor Spider eval scripts, write `src/` modules + CPU tests, upload `sqlbuddy-src` dataset, push `01_baseline_eval` with N_EXAMPLES=50.

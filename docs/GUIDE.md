# SQLBuddy: project guide for the group

One page that answers four questions:
1. What has been done?
2. What ran in the cloud, and what runs locally?
3. What should you check?
4. How do you evaluate?

Details live in the files linked at each point.

---

## 1. What has been done (all 7 phases)

| Phase | What | Result / where |
|---|---|---|
| 0 | Environment check, repo scaffolding, `.gitignore` for tokens/weights/data | — |
| 1 | Data pipeline + **base model evaluation** on Spider dev (1,034 questions, 20 unseen DBs) | EX 57.0 %, `results/baseline/` |
| 2 | **LoRA fine-tuning** of Qwen2.5-Coder-1.5B-Instruct (r 16, 2 epochs, 820 steps, 2 h 10 min on a T4) | adapter on Hugging Face (public), `results/training/` |
| 3 | **Fine-tuned evaluation** on the same 1,034 questions, same prompt and decoding | EX 73.1 %, `results/finetuned/` |
| 4 | **Comparison and analysis**: McNemar test, bootstrap CI, charts, 50 failures labelled by hand | `results/comparison.csv`, `statistics.json`, `error_analysis.md` |
| 5 | **Demo**: Gradio app (read-only, SELECT only), sample `university.db`, Kaggle GPU notebook | `app/`, `sample_db/`, `notebooks/04_demo` |
| 6 | **Documentation**: README, presentation notes | `README.md`, `PRESENTATION_NOTES.md` |
| 7 | **Demo redesign** + **latency study** + **extra evaluation** (per database, out-of-domain) + **slide deck** | `docs/PERFORMANCE.md`, `results/per_database.csv`, `results/ood/` |

### Headline results (Spider dev, 1,034 held-out questions)

| | Execution accuracy | Exact match | Valid SQL |
|---|---|---|---|
| Base model | 57.0 % | 32.4 % | 76.6 % |
| Fine-tuned (LoRA) | **73.1 %** | **68.0 %** | **91.8 %** |

- Fine-tuning fixed 220 questions and broke 53. Exact McNemar test: p = 2.4 × 10⁻²⁵. 95 % CI of the gain: +13.2 to +19.1 points.
- 18 of the 20 unseen databases improve.
- **Out-of-domain** (30 questions we wrote for `university.db` before running either model): base 36.7 %, fine-tuned 73.3 %; 11 fixed, 0 broken; p = 0.001.

---

## 2. What was done in the cloud vs. what runs locally

### Cloud: Kaggle (GPU) and Hugging Face. Already done; you don't need to repeat it.

All GPU work ran on a free **Kaggle T4** through the `kaggle` CLI. Every run is logged in
[`docs/KAGGLE_RUNS.md`](KAGGLE_RUNS.md) (10 runs, including failed ones and their fixes).

| Kaggle notebook | Purpose | Output |
|---|---|---|
| `sqlbuddy-01-baseline-eval` | base model on 1,034 dev questions (~6 min) | `results/baseline/` |
| `sqlbuddy-02-train-lora` | LoRA training (~2 h 10 min) | adapter + training logs |
| `sqlbuddy-03-finetuned-eval` | fine-tuned model on the same 1,034 questions (~5 min) | `results/finetuned/` |
| `sqlbuddy-04-demo` | **the demo for the presentation**: the same app on a GPU, with a public `gradio.live` link | link printed in the last cell |

**Supporting services:**
- **Kaggle dataset `sqlbuddy-src`** carries our `src/` and `app/` code into the notebooks.
- **Kaggle dataset `yale-universitys-spider-10-nlp-dataset`** provides the Spider SQLite databases.
- **Hugging Face, public repo `ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora`**, holds the trained
  adapter (74 MB). Anyone can download it without a login. Weights are never stored in git.
- **GitHub `main`** is the source of the demo code. Notebook 04 downloads the app from GitHub at the start of
  every run, so the Kaggle demo always shows the latest merged design.

### Showing the app without installing anything: Kaggle notebook 04
1. Log in at kaggle.com, then open **Your Work → Code → `sqlbuddy-04-demo` → Edit**.
2. In the right panel, set **GPU T4** and **Internet on**.
3. Click **Run all**. After 3–5 minutes the last cell prints `PUBLIC LINK: https://….gradio.live`.
4. The link works for anyone for **90 minutes**, at about 1–4 s per question. Press **Stop** when finished.
   Kaggle gives about 30 free GPU hours per week.

Hugging Face Spaces was considered, but running a Gradio app on its free CPU now needs a paid PRO
subscription, so Kaggle is our free hosting.

### Locally: CPU only, no GPU needed

| What | Command | Time |
|---|---|---|
| The demo app | `python app/app.py`, then open http://127.0.0.1:7860 | first start downloads ~3 GB; then about 3–15 s per question |
| Unit tests (56) | `SPIDER_DIR=<spider folder> python -m pytest -q tests` | ~15 s |
| Comparison + statistics | `python -m src.compare results/baseline results/finetuned results` | seconds |
| Charts + error tables + per-DB | `python -m src.analysis results/baseline results/finetuned results/training results` | seconds |
| Out-of-domain test | `python -m src.ood_eval` | ~15 min on a 4-core CPU |
| **After changing the app** | merge into `main`, then `bash scripts/deploy.sh` | ~5 min, including a Kaggle test run |

Install and run the demo on a laptop:
- The full commands for Windows and macOS are in [README §7](../README.md#7-run-the-demo-on-your-laptop-windows--macos).
- In short: create a venv, then `pip install -r requirements-demo.txt` (on Windows, install CPU torch first),
  then `python app/app.py`. No Hugging Face login is needed: the adapter is public.
- **Why it is slower than on Kaggle.** The CPU runs the whole model once per generated token (~0.16 s per
  token). See [`docs/PERFORMANCE.md`](PERFORMANCE.md).

---

## 3. What to check (checklist)

**Demo (before the presentation)**
- [ ] The app (Kaggle link, or `python app/app.py` locally) shows **Model ready** and **Fine-tuned adapter loaded**.
      If it says *adapter unavailable*, the internet connection to Hugging Face failed.
- [ ] Choose **Compare both** and ask *"What is the average score in each course? Show the course title."*
  - Base should show **SQL error** (`no such column: T2.title`).
  - Fine-tuned should show **Done · 12 row(s)**.
- [ ] Each panel shows the device, the mode, the token count and the generation time.
- [ ] **Stop** cancels a running question; **Generate** is greyed out while one runs.
- [ ] **Upload my own** works with a `.sqlite` file (try a Spider DB such as `concert_singer.sqlite`).
- [ ] The safety check rejects writes. To test it, run this in Python from the repo folder:
  ```python
  from app.sqlbuddy_core import run_select, SAMPLE_DB
  run_select(SAMPLE_DB, "DROP TABLE students")
  ```
  It should raise *only SELECT queries are allowed*.

**Results (for the report)**
- [ ] `results/comparison.csv` matches the table above. Every number in the README and slides comes from `results/`.
- [ ] `results/statistics.json` holds the McNemar p-values and confidence intervals per level.
- [ ] `results/error_analysis.md` is the written analysis: fixes, 50 hand-labelled failures, regressions, the out-of-domain section and limitations.
- [ ] Charts:
  - `results/exec_acc_by_level.png`
  - `results/exact_match_by_level.png`
  - `results/valid_sql_rate.png`
  - `results/loss_curve.png`
  - `results/exec_acc_by_database.png`

**Presentation**
- [ ] Slide deck: https://claude.ai/artifact/GVSZ3uR7kogY9rAMugzxJH. The cover is filled in (names, roll numbers, CS4063). Share it from its Share menu, or others can't open it.
- [ ] [`PRESENTATION_NOTES.md`](../PRESENTATION_NOTES.md): slide-by-slide notes, a demo script and backup Q&A.

---

## 4. How to evaluate (and how to read the numbers)

### The protocol (fixed before testing)
- **Test set:** Spider dev, 1,034 questions on 20 databases never used in training. A validation split of
  8 *whole* training databases was used only to watch the loss.
- **Same everything except fine-tuning:**
  - the same prompt (schema as `CREATE TABLE` + question → "SQL only");
  - greedy decoding with at most 256 new tokens;
  - the same SQL extraction.
- **Metrics,** each broken down by official Spider difficulty (easy / medium / hard / extra):
  - **Execution accuracy (primary):** run the predicted and gold SQL and compare the result rows. Order only
    matters if the gold query has `ORDER BY`.
  - **Exact match:** the official Spider exact-set-match, with values ignored.
  - **Valid-SQL rate:** the prediction runs without an error.
- **Significance:** both models answer the same questions, so the comparison is paired:
  - exact **McNemar test** on the flipped questions;
  - **paired bootstrap** 95 % CI of the gain.

### Re-running the evaluation
- **Statistics, charts and error tables from the saved predictions (CPU, seconds):** run `src.compare`, then
  `src.analysis` (commands in section 2).
- **Out-of-domain test (CPU, ~15 min):** `python -m src.ood_eval`. Results go to `results/ood/`.
- **The full GPU evaluation again (Kaggle):** follow [README §8](../README.md#8-reproduce-the-experiments-kaggle).
  - Upload the code with `bash scripts/upload_src.sh "msg"`.
  - Build the notebooks with `python notebooks/build_notebooks.py`.
  - Run `kaggle kernels push -p notebooks/01_baseline_eval --accelerator NvidiaTeslaT4`, and the same for
    `03_finetuned_eval`.
  - Download the outputs with `kaggle kernels output …`.

### How to interpret
- **Use execution accuracy as the main number.** Exact match rises more (+35.6 points) partly because the
  model learned Spider's writing style.
- **Execution accuracy uses one database instance per question.** It can accept a query that is right by
  coincidence, or reject a correct one because of a type difference such as `'3'` vs `3`. Spider's
  test-suite accuracy would be stricter.
- **Remaining weakness: schema linking.** 27 of the 50 reviewed failures choose the wrong table, join or
  column, and 5 fail because the model never sees cell values.
- **Training loss.** Validation loss rose in epoch 2; the best was at step 200. We tested the pre-registered
  final adapter. Early stopping is future work.

---

## Where everything is (map of the Markdown files)

| File | What it is for |
|---|---|
| `docs/GUIDE.md` | **this page**: overview, cloud vs local, checklist, evaluation |
| `README.md` | full project report: method, results, charts, limitations, laptop demo, Kaggle reproduction |
| `PRESENTATION_NOTES.md` | slide-by-slide outline, demo script, backup Q&A |
| `results/error_analysis.md` | detailed error analysis (fixes, failures, regressions, out-of-domain) |
| `docs/PERFORMANCE.md` | why the demo takes seconds per question on a CPU, and what was optimised |
| `docs/PLAN.md` | the phases with their checklists |
| `docs/PROGRESS.md` | running log, newest first, with next steps |
| `docs/DECISIONS.md` | every design decision with reasons and alternatives |
| `docs/KAGGLE_RUNS.md` | every Kaggle GPU run, including failures and fixes |
| `CLAUDE.md` | short project brief for the AI assistant (start of each session) |

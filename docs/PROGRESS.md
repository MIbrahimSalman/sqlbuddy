# Progress log (newest first)

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

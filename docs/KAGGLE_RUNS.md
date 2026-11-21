# Kaggle runs

Kaggle user: `mibrahimsalman022`. All kernels private, GPU T4.

| # | Notebook | Date | Settings | Run time | Status | Outcome | Link |
|---|----------|------|----------|----------|--------|---------|------|
| 1 | 01_baseline_eval v1 | 2026-11-21 | base model, N=50 (smoke), bs 16, greedy, 256 tok; torch 2.11.0+cu130, transformers 5.22.1 | 2.0 min (gen 12 s) | complete | smoke OK; EX 54% on first 50; invalid preds are genuine model errors (wrong alias) | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-01-baseline-eval (v1) |
| 2 | 01_baseline_eval v2 | 2026-11-21 | base model, full dev N=1034, same settings | 6.0 min (gen 224 s, score 26 s) | complete | **EX 56.96 / EM 32.40 / valid 76.60** → `results/baseline/` | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-01-baseline-eval (v2) |

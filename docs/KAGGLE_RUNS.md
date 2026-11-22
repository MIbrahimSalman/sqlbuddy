# Kaggle runs

Kaggle user: `mibrahimsalman022`. All kernels private, GPU T4.

| # | Notebook | Date | Settings | Run time | Status | Outcome | Link |
|---|----------|------|----------|----------|--------|---------|------|
| 1 | 01_baseline_eval v1 | 2026-11-21 | base model, N=50 (smoke), bs 16, greedy, 256 tok; torch 2.11.0+cu130, transformers 5.22.1 | 2.0 min (gen 12 s) | complete | smoke OK; EX 54% on first 50; invalid preds are genuine model errors (wrong alias) | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-01-baseline-eval (v1) |
| 2 | 01_baseline_eval v2 | 2026-11-21 | base model, full dev N=1034, same settings | 6.0 min (gen 224 s, score 26 s) | complete | **EX 56.96 / EM 32.40 / valid 76.60** → `results/baseline/` | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-01-baseline-eval (v2) |
| 3 | 02_train_lora v1 | 2026-11-21 | smoke, MAX_STEPS=30 | 2.8 min | **error** | `ImportError: incompatible torchao 0.10.0` (Kaggle image) raised by PEFT when creating LoRA layers → fix: `pip uninstall -y torchao` in notebooks 02/03 | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-02-train-lora (v1) |
| 4 | 02_train_lora v2 | 2026-11-21 | smoke, same | 0.3 min | **error** | `SyntaxError`: two-line cell `pip uninstall…\npip install…` — IPython auto-magic only works for single-line cells → use explicit `%pip`; added a local IPython parse check of all notebook cells | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-02-train-lora (v2) |
| 5 | 02_train_lora v3 | 2026-11-21 | smoke, MAX_STEPS=30, eval every 30 steps, bs 4×4, max_len 1536 | 8.7 min (train 329 s) | complete | 10.96 s/step, peak GPU mem 5.41 GB, train loss 0.221→0.203, val loss 0.197; 6,560 train / 352 val (88 dropped > 1536) → full run est. ~2.5 h | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-02-train-lora (v3) |
| 6 | 02_train_lora v4 | 2026-11-22 | **full**: 2 epochs (~820 steps), eval/ckpt every 100 steps | 2 h 10 min training (9.49 s/step, 820 steps) | ✅ complete | train loss 0.255→0.036; val loss min 0.184 @ step 200, final 0.223; adapter → HF | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-02-train-lora (v4) |
| 7 | 03_finetuned_eval v1 | 2026-11-22 | smoke: 50 dev questions with the adapter | ~3 min (gen 12 s) | complete | EX 76.0 / EM 76.0 / valid 88.0 on first 50 (base on same 50: EX 56 / EM 42); EX≠EM on 2 rows, columns independent | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-03-finetuned-eval (v1) |
| 8 | 03_finetuned_eval v2 | 2026-11-22 | **full**: 1,034 dev questions with the adapter | ~5 min (gen 223 s) | ✅ complete | EX 73.11 / EM 67.99 / valid 91.78 (base 56.96 / 32.40 / 76.60) | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-03-finetuned-eval (v2) |
| 9 | 04_demo v1 | 2026-11-22 | smoke: self-test (base + FT on one question) + share link, DEMO_MINUTES=1 | – | running | – | https://www.kaggle.com/code/mibrahimsalman022/sqlbuddy-04-demo (v1) |

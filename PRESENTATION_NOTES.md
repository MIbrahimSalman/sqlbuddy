# SQLBuddy: presentation notes (15 slides, about 15 min + demo)

**Slide deck:** https://claude.ai/artifact/GVSZ3uR7kogY9rAMugzxJH (private until shared from its Share menu;
it can be downloaded as PowerPoint or PDF). Speaker notes are attached to every slide.

Before presenting:
- The cover lists Fatima Malik (23L-0826), Mohammad Ibrahim Salman (23L-0723) and CS4063 – Natural Language Processing (no date, as requested).
- About 5 minutes before presenting, start the Kaggle demo: kaggle.com → `sqlbuddy-04-demo` → Edit → GPU T4 + Internet on → Run all, then open the printed `PUBLIC LINK` (it stays up 90 min). Backup: `python app/app.py` on a laptop.

Every number below comes from `results/`.

| # | Slide | Key message / numbers | Visual |
|---|---|---|---|
| 1 | Title | Base 57.0 % → fine-tuned 73.1 % execution accuracy on unseen databases | big numbers |
| 2 | Problem | Text-to-SQL = semantic parsing; can a laptop-sized model do it? | question → SQL example |
| 3 | Research question | Controlled comparison: same model, prompt, decoding; held-out test; settings fixed in advance | 4 cards |
| 4 | Data | Train 6,560 (132 DBs) / val 352 (8 held-out DBs) / test 1,034 (20 unseen DBs) / out-of-domain 30 | table + difficulty tags |
| 5 | Method | One pipeline: schema → prompt → model → greedy decode → execute | flow diagram |
| 6 | LoRA | 18.5 M trainable params (1.2 %), 2 epochs, 2 h 10 min on a T4; val loss best at step 200 | `loss_curve.png` |
| 7 | Evaluation | Execution accuracy (primary), exact match, valid SQL; McNemar + bootstrap | cards |
| 8 | Results | EX 57.0 → 73.1 (+16.2), EM 32.4 → 68.0, valid 76.6 → 91.8; hard +24.1 | `exec_acc_by_level.png` |
| 9 | Significance | 220 fixed vs 53 broken; p = 2.4 × 10⁻²⁵; CI [+13.2, +19.1] | 2×2 flip table |
| 10 | Per database | 18 of 20 DBs improve; largest +43.3 (course_teach), only drop −2.3 (pets_1) | `exec_acc_by_database.png` |
| 11 | Out-of-domain | Own university.db, 30 questions: 36.7 → 73.3, 11 fixed / 0 broken, p = 0.001 | big numbers + table |
| 12 | Error analysis | Alias errors 201 → 61; EM rises more than EX because of style; 27/50 remaining failures are schema linking | code examples + table |
| 13 | Demo | Live Compare mode on a free Kaggle T4 GPU (~1–4 s per question, public link, always the latest app from GitHub); read-only safety; laptop CPU fallback 3–15 s | `demo_screenshot.png` |
| 14 | Limitations | One DB instance, contamination, epoch-2 over-fitting, schema only, single run, CPU speed | 6 cards |
| 15 | Conclusion | +16.2 points, 18/20 DBs, 1.2 % of weights; remaining challenge = schema understanding | 3 numbers |

## Demo script (slide 13)

1. Show the status bar: device, **Model ready**, adapter loaded.
2. Pick **Compare both** and the example *"What is the average score in each course? Show the course title."*
   - The base model fails with `no such column: T2.title`; the app explains it as a schema-linking error.
   - The fine-tuned model returns 12 rows.
   - Point at the measured time and the token count.
3. Optional:
   - Upload a Spider database, e.g. `concert_singer.sqlite`, and ask *"How many singers are there?"*
   - Show that `DROP TABLE` is blocked.
4. If the laptop is slow:
   - Use short questions such as *"How many students are there?"*, about 3 s.
   - Or use the Kaggle GPU notebook `04_demo` (public link).
   - Or show the screenshot.

## Backup Q&A

- **Why LoRA and not full fine-tuning?**
  - A full fine-tune of 1.5 B parameters with AdamW needs about 25 GB of GPU memory; LoRA needed 5.4 GB on a T4.
  - It also limits catastrophic forgetting, and the adapter is only 74 MB.
- **Why a database-disjoint validation split?** Test databases are unseen, so validation databases must be
  too; otherwise validation loss would be optimistic.
- **Why execution accuracy as the primary metric?** Different SQL can be equally correct. Exact match punishes
  valid alternatives and rewards the annotators' style (slide 12).
- **Why McNemar and not a t-test or chi-square on two proportions?** The two models answer the *same*
  questions, so the samples are paired. Only the flipped questions carry information.
- **Isn't the base model at a disadvantage on a Spider-style prompt?**
  - Both models get the identical prompt.
  - On our own database, which is not in Spider style, the gap is even larger (slide 11).
- **Why does validation loss rise if accuracy improves?**
  - Loss measures token-level confidence; execution accuracy measures whether the final answer is right.
  - The model becomes over-confident on some tokens without getting more answers wrong overall.
  - Early stopping on validation execution accuracy is future work.
- **Why is it slow on a laptop?** The whole network runs once per generated token, about 0.16 s per token
  on 4 CPU cores. It stops at the end of the SQL, so no time is wasted. See `docs/PERFORMANCE.md`.
- **Is the demo safe on a real database?** It uses a read-only connection, a single SELECT only, an SQLite
  authorizer that allows only reads, a 10 s limit and 200 rows at most.

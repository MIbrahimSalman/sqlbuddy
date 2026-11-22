# SQLBuddy: presentation outline (10 slides, about 12–15 min)

Every number comes from `results/` and `docs/`. The chart for each slide is named in *italics*.

---

### 1. Title and the one-sentence result
**SQLBuddy: asking databases in plain English with a fine-tuned small language model**
- Claim: LoRA fine-tuning a 1.5 B model raises execution accuracy on unseen databases from **57.0 % to 73.1 %**.
- Group names, course, date.
- *Speaker note:* say the result first, then explain how we got it and why we trust it.

### 2. Problem and motivation
- People need data but can't write SQL. Text-to-SQL is semantic parsing: question + schema → executable query.
- Large API models work but are costly and send your schema off-site. Can a **laptop-sized** model do it?
- Research question: *does parameter-efficient fine-tuning measurably beat the same base model on held-out data?*

### 3. Dataset: Spider
- 200 databases in 138 domains, human-written question/SQL pairs, cross-domain by design.
- Train: 6,560 pairs from 132 DBs (`train_spider.json` has 140; 8 are held out).
- Validation: 352 pairs from 8 held-out DBs.
- **Test: Spider dev, 1,034 questions from 20 DBs never seen in training.**
- Difficulty levels: easy 248, medium 446, hard 174, extra 166.
- Show one example: question → schema → gold SQL.

### 4. Method: model and prompt
- Base model: Qwen2.5-Coder-1.5B-Instruct (code-pretrained, small enough for a CPU).
- Prompt: system instruction + schema as `CREATE TABLE` (with PK/FK) + question → "SQL only".
  - *The same prompt for both models*, so the only variable is the fine-tuning.
- Show the pipeline diagram (mermaid in the README).

### 5. Fine-tuning with LoRA
- LoRA: freeze the 1.56 B weights and learn low-rank updates ΔW = BA (r = 16) on attention + MLP.
  - That is **18.5 M trainable parameters (1.2 %)**, a 74 MB adapter.
- Loss only on SQL tokens (prompt masked).
- Training setup: 2 epochs, lr 2e-4 cosine, effective batch 16, fp16.
  - Free Kaggle T4: **2 h 10 min**, 5.4 GB GPU memory.
- *loss_curve.png*: training loss 0.26 → 0.04. Validation loss is best at step 200, then rises.
  - Mention this honestly: it is over-fitting at the token level, and we come back to it on slide 9.

### 6. Evaluation methodology
- **Execution accuracy (primary):** run predicted and gold SQL and compare the result rows.
- **Exact match:** the official Spider component match, values ignored.
- **Valid-SQL rate:** the query runs without error.
- Results broken down by difficulty.
- Paired statistics: same 1,034 questions → McNemar test + bootstrap CI.
- Every setting was fixed *before* testing; the test set was never used for any choice.

### 7. Results
- *exec_acc_by_level.png*: EX **57.0 → 73.1 (+16.2 pp)**, gains at every level.
  - Hard improves most: +24.1.
  - Extra improves least: +9.6.
- Exact match 32.4 → 68.0; valid SQL 76.6 → 91.8 (*valid_sql_rate.png*).
- Flip table: **220 fixed vs. 53 broken**, McNemar p = 2.4 × 10⁻²⁵, 95 % CI of the gain [+13.2, +19.1] pp.
- *Speaker note:* the improvement is not luck. The confidence interval stays far away from zero.

### 8. Error analysis: what changed and what is still hard
- Fixes come from **broken SQL disappearing**.
  - Wrong-alias errors (`no such column: T2.title`): 201 → 61.
  - 62 % of the fixes were base queries that did not even run.
- The model learned Spider's **style**: no `INNER JOIN`, about as many joins as the gold queries.
  - This explains why EM jumps more than EX.
- 50 failures reviewed by hand:
  - **schema linking** (wrong table, join key or column): 27/50;
  - nested or set logic: 6;
  - unseen cell values (`'French'` vs `'France'`): 5;
  - gold or evaluation errors: 2.
- Show 1 fixed example (`test_515`, alias) and 1 still-wrong example (`test_115`, `country = 'USA'` but country is an id).

### 9. Limitations and future work
- One DB instance per question, so EX can accept a coincidentally correct query. Test-suite accuracy would be stricter.
- Spider is public, so contamination is possible. It affects both models equally, so the *difference* stays fair.
- Validation loss rose in epoch 2. Future work: early stopping or checkpoint selection on validation EX.
- Future work on the prompt and data:
  - add sample cell values to the prompt;
  - test on harder benchmarks (BIRD, Spider 2.0);
  - try other SQL dialects;
  - run multiple seeds.

### 10. Live demo and conclusion
- Gradio app: choose a DB (sample `university.db` or upload your own), see the schema, ask a question, **Compare both**.
- Demo question: *"What is the average score in each course? Show the course title."*
  - Base fails with `no such column: T2.title`.
  - Fine-tuned returns 12 rows.
- Safety: read-only connection, SELECT-only, authorizer, timeout.
- Runs on a laptop CPU (about 3–15 s per question). Backup: Kaggle GPU notebook with a public link (`04_demo`).
- **Conclusion:** cheap LoRA fine-tuning (1.2 % of the weights, about 2 GPU-hours) gives a small open model
  a large, statistically solid improvement in text-to-SQL. The remaining problem is understanding the
  schema, not SQL syntax.

---

**Backup slides / Q&A**
- Why LoRA and not full fine-tuning? Memory: a full 1.5 B fine-tune with AdamW needs about 25 GB; LoRA needs 5.4 GB on a T4. It also avoids catastrophic forgetting.
- Why a database-disjoint validation split? Test databases are unseen, so validation must be too. Otherwise the validation loss would be too optimistic.
- Why execution accuracy as the primary metric? Different SQL can be equally correct. EM punishes valid alternatives and rewards style (see slide 8).
- Demo fallback if Wi-Fi fails: `results/demo_screenshot.png`.

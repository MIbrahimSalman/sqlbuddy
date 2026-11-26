# Error analysis — base vs. LoRA fine-tuned Qwen2.5-Coder-1.5B-Instruct

Test set: Spider dev, 1,034 questions on 20 databases, never used for training or model selection.
Primary metric: execution accuracy (EX). All numbers come from `results/finetuned/predictions.csv` and
`results/baseline/predictions.csv`, produced by `src/compare.py` and `src/analysis.py`.

## 1. Headline result

| Level (n) | EX base | EX fine-tuned | Δ (pp) | 95 % CI of Δ | EM base → FT | Valid SQL base → FT |
|---|---|---|---|---|---|---|
| easy (248)   | 72.18 | **88.31** | +16.13 | [10.9, 21.4] | 63.31 → 88.71 | 89.11 → 98.79 |
| medium (446) | 59.19 | **74.66** | +15.47 | [10.8, 20.4] | 27.13 → 67.94 | 74.89 → 91.03 |
| hard (174)   | 39.66 | **63.79** | +24.14 | [16.7, 31.6] | 16.09 → 54.02 | 70.69 → 92.53 |
| extra (166)  | 46.39 | **56.02** | +9.64  | [3.0, 16.9]  | 17.47 → 51.81 | 68.67 → 82.53 |
| **all (1,034)** | **56.96** | **73.11** | **+16.15** | **[13.2, 19.1]** | 32.40 → 67.99 | 76.60 → 91.78 |

The CI is a paired bootstrap with 10,000 resamples.

## 2. Is the improvement real? Flips and McNemar's test

Both models answer the *same* 1,034 questions, so the comparison is paired. The question is how many
answers flipped.

| | FT correct | FT wrong |
|---|---|---|
| **Base correct** | 536 | **53** (regressions) |
| **Base wrong**   | **220** (fixes) | 225 |

- Net gain: 220 − 53 = **+167 questions**.
- **Exact McNemar test** (binomial test on the 273 discordant pairs; H0: a flip is equally likely in
  either direction): **p = 2.4 × 10⁻²⁵**.
- The improvement is significant at every difficulty level:
  - easy: p = 4e-9.
  - medium: p = 1e-9.
  - hard: p = 3e-9.
  - extra: p = 0.014.
- Per-level numbers are in `results/statistics.json` and every flipped question is in `results/flips.csv`.

## 3. What fine-tuning fixed

### 3.1 Base-model failures, all 445 wrong answers, automatic categories

These categories come from `src/analysis.py` (sqlglot structural comparison against gold) and are in
`results/error_types.csv`.

| Category | base | fine-tuned |
|---|---|---|
| invalid SQL (does not execute) | **242** (54 %) | **85** (31 %) |
| wrong table | 72 | 58 |
| nested query / set operation | 52 | 29 |
| wrong column | 34 | 47 |
| GROUP BY / aggregation | 25 | 34 |
| wrong condition / value | 11 | 17 |
| missing / incorrect JOIN | 6 | 5 |
| ORDER BY / LIMIT | 3 | 3 |
| **total failures** | **445** | **278** |

### 3.2 The main effect: alias errors

- **Main effect: the model stops producing broken SQL.** Of the 220 questions fine-tuning fixed, 137 (62 %)
  were base predictions that did not even execute.
- **The dominant base error is a wrong table alias:** `no such column: T2.x`, where the column exists
  but on the other joined table.
  - Base model: 201 of these errors.
  - Fine-tuned model: 61 (−70 %).
  - "no such table" errors: 15 → 3.

### 3.3 Fine-tuning taught the dataset's SQL *style*

| | base | fine-tuned | gold |
|---|---|---|---|
| `INNER JOIN` used | 40.1 % of queries | 0 % | (Spider writes `JOIN`) |
| JOINs per query | 0.65 | 0.54 | 0.50 |
| mean SQL length (chars) | 119 | 106 | 107 |
| raw output ≠ extracted SQL (only line breaks / trailing `;`, never extra text or fences) | 13.0 % | 0 % | – |

- The base model over-joins: it adds tables "to be safe", and that is where its alias errors come from.
- The fine-tuned model writes queries as compact as the annotators' gold SQL.
- This is the main reason why **exact match jumps much more (+35.6 pp) than execution accuracy (+16.2 pp)**:
  - Exact match rewards the Spider query structure.
  - Part of the EM gain is learned *annotation style*, not new reasoning ability.
  - EX is therefore the more honest metric.

### 3.4 Five examples fixed by fine-tuning

1. **Alias error removed** (`test_515`, extra). *Which department offers the most number of degrees?*
   - Base: `SELECT T2.department_name ... FROM departments AS T1 JOIN degree_programs AS T2 ...` → `no such column`
     (department_name is on T1).
   - FT: `SELECT T2.department_name, T1.department_id FROM degree_programs AS T1 JOIN departments AS T2 ...
     GROUP BY T1.department_id ORDER BY count(*) DESC LIMIT 1` ✔
2. **Unnecessary join removed** (`test_473`, medium). *Find the number of players for each country.*
   - Base joins `rankings`, which multiplies each player by their ranking rows, so the count is wrong.
   - FT: `SELECT count(*), country_code FROM players GROUP BY country_code` ✔
3. **Right column chosen** (`test_807`, easy). *Count the number of countries in Asia.*
   - Base: `WHERE region = 'Asia'` (0 rows).
   - FT: `WHERE continent = 'Asia'` ✔
4. **Set operation instead of an invented condition** (`test_32`, hard). *Stadiums that did not have a concert in 2014?*
   - Base: a NOT IN with an invented `Singer_ID IS NULL` filter.
   - FT: `SELECT name FROM stadium EXCEPT SELECT T1.name FROM stadium AS T1 JOIN concert AS T2 ... WHERE T2.year = 2014` ✔
5. **Aggregation repaired** (`test_850`, medium). *Record formats of orchestras in ascending order of count.*
   - Base: `SELECT DISTINCT ... ORDER BY COUNT(*)` → `misuse of aggregate`.
   - FT: `SELECT major_record_format FROM orchestra GROUP BY major_record_format ORDER BY count(*) ASC` ✔

**Caveat: EX can accept a wrong query.** `test_197` (*Which city and country is the Alton airport at?*)
counts as "fixed". But the FT query filters on `FlightNo = 30001` instead of `AirportName = 'Alton'`, and it
returns the right row only by coincidence. EX is an upper bound on true accuracy.

## 4. What is still wrong: manual review of 50 fine-tuned failures

The 50 failures are a stratified random sample (seed 42; easy 5, medium 20, hard 11, extra 14). Each was
read and labelled by hand; see `results/error_sample.csv` (columns `manual_category`, `note`).

| Manual category | n | typical example |
|---|---|---|
| schema linking: wrong table / join path | **16** | joins on a meaningless key (`car_makers.Id = cars_data.id`), or misses that `countries.Continent` is an id that needs a join |
| nested query / set-operation logic | 6 | `INTERSECT` rewritten as a join; per-row filter instead of `NOT IN (… HAVING sum > 1000)` |
| schema linking: wrong column | 5 | `Region` instead of `Continent`, `gnp` instead of `LifeExpectancy` |
| value format (case / encoding) | 5 | `'Yes'` vs stored `'yes'`, `'left'` vs stored `'L'`, `'French'` vs `'France'` |
| aggregation / sort semantics | 4 | "oldest player" sorted by birth date DESC; "best rank" as `max` |
| wrong table alias | 4 | still some `T3.document_name` errors after spurious joins |
| missing condition | 3 | 3 × forgot `IsOfficial = 'T'` for "official language" |
| DISTINCT / duplicate rows | 3 | adds or omits DISTINCT where the gold does the opposite |
| hallucinated column name | 2 | `petage` for `pet_age`, `pet_type` for `pettype` |
| gold error or evaluation artifact | 2 | gold uses `max(age)` for "youngest"; `'3'` (text) vs `3` (int) |

### Findings

- **Schema linking is now the bottleneck** (16 + 5 + 4 + 2 = 27 of 50). The model knows *how* to write SQL,
  but it often picks the wrong table, join key or column. This is worst on databases with
  id-coded foreign keys (`car_1`, `dog_kennels`, `world_1`).
- **The model never sees cell values.** Five failures are value-format mismatches that no schema-only
  prompt can solve. Adding 2–3 sample rows per table to the prompt is the obvious next step.
- **Nested logic remains hard**, which is why extra-hard has the smallest gain (+9.6 pp).
- **2 of 50 (4 %) are not real model errors.** EX slightly *underestimates* in these cases, and test_197
  shows it can also *overestimate*.

### Five still-wrong examples

1. `test_115` (hard). *How many car models are produced in the usa?*
   - FT: `... WHERE country = 'USA'`, but `car_makers.country` is a numeric id; it needs a join to `countries`.
2. `test_5` (medium). *Average, minimum and maximum age for all French singers?*
   - FT: `WHERE country = 'French'`; the stored value is `'France'`. This is a value problem.
3. `test_901` (hard). *High schoolers who both have friends and are liked?*
   - FT joins `Friend` and `Likes` directly, which produces duplicates; gold uses `INTERSECT`.
4. `test_768` (medium). *Official language of the country whose head of state is Beatrix?*
   - Correct join, but `IsOfficial = 'T'` is missing, so non-official languages are returned too.
5. `test_34` (medium). *Names, themes and number of singers for every concert?*
   - FT groups by `concert_name`, and two concerts are both called "Week 1". Grouping by the key would be correct.

## 5. Regressions (base correct → fine-tuned wrong): 53 questions

About 30 of the 53 regressions were read in detail; the full list is in `results/flips.csv` (`flip = correct_to_wrong`).

| Pattern | Examples |
|---|---|
| **Over-simplification**: drops a join that *was* needed | `test_710`/`711` *Which region is Kabul in?* → `SELECT district FROM city` / treats Kabul as a country; `test_805` reads `city` instead of `country` |
| **Wrong column on a familiar DB** | `test_801` orders by `population` instead of `SurfaceArea`; `test_762` returns `gnp` |
| **Sort direction / semantics** | `test_453`/`454` "oldest player" → `birth_date DESC` (base was right) |
| **Spider-style value bias** | `test_485` `hand = "left"` (base used `'L'`); `test_182` `"Jetblue Airways"` vs stored `"JetBlue Airways"` |
| **Wrong GROUP BY key** | `test_886`, `test_910` group by `friend_id` instead of `student_id` |
| **DISTINCT changes** | `test_867` adds DISTINCT, `test_781` drops it |

### Interpretation

Fine-tuning shifted the model toward *short* queries. That removed far more broken over-joined queries
(220 fixes) than it created under-joined ones (53 regressions). However, a few regressions on `world_1`
(Kabul, Aruba, region vs continent) suggest the model is sometimes overconfident. This matches the training
log, where validation loss rose in epoch 2 (best at step 200: 0.184; final: 0.223). An earlier checkpoint
or early stopping on validation EX might keep the fixes and avoid some regressions. That is left as future
work, because only the final adapter was pre-registered for testing.

## 6. Limitations of this evaluation

- **Execution accuracy runs on one database instance per question.**
  - It can give false positives: `test_197` is a coincidental match.
  - It can give false negatives: type mismatches like `'3'` vs `3`.
  - Spider's test-suite accuracy (several DB instances per question) would be stricter.
- **Exact match ignores values and is sensitive to style.** Its large jump partly measures learned
  annotation style; see 3.3.
- **Possible contamination.** Spider is public and the base model already writes Spider-style SQL (e.g. `T1/T2`
  aliases). Its pre-training data may contain Spider. This affects both models equally, so the *difference*
  is still a fair estimate of the fine-tuning effect.
- **Model selection.**
  - The validation split (8 held-out train DBs) was only used to monitor loss.
  - No hyper-parameter search was run on the test set.
  - The plan fixed every setting before testing.

## 7. Out-of-domain check (30 questions on our own university.db)

Results are in `results/ood/`. The 30 questions and gold SQL were fixed before running the models.

- **Result.**
  - Base: 36.7 % execution accuracy; fine-tuned: 73.3 %.
  - 11 questions fixed, 0 broken; exact McNemar p = 0.001.
- **Same error pattern as on Spider, only stronger.** In 13 of the base model's 19 failures it joins an
  extra table and then reads a column from the wrong alias (`SELECT T2.name FROM students AS T1 JOIN
  enrollments AS T2 …`).
- **The fine-tuned model's 8 failures are schema linking again:**
  - an invented `departments` table;
  - `enrollment_year` placed on `enrollments`;
  - joining `grades.enrollment_id` to `students.student_id`;
  - one value error: `semester = "Fall"` instead of `'Fall 2025'`.
- **Takeaway.** The improvement is not specific to Spider's databases. The remaining weakness, linking the
  question to the right tables and columns, is the same everywhere.


# SQLBuddy: presentation script and Q&A (10 slides)

**Deck:** https://claude.ai/artifact/GVSZ3uR7kogY9rAMugzxJH (export: Share → Export → PDF/PowerPoint).
**Team:** Fatima Malik (23L-0826), Mohammad Ibrahim Salman (23L-0723). **Course:** CS4063 Natural Language Processing.

**Total time:** about **14–15 minutes** of talking (about 2,250 spoken words at a normal pace of ~155 words per
minute), plus 1–2 minutes of live demo after slide 8, then questions.
- **Running long?** Skip the "Following one question through" paragraph on slide 5 and the car-models example on
  slide 9. That saves about 1 minute.
- **Running short?** On slide 4, also explain the record's fields (db_id, question, query, tables.json, the
  .sqlite file) from the "Show one record" box below its script. That adds about 30 seconds.

**Suggested split:**
- Fatima: slides 1–5 (problem, tools, dataset, pipeline).
- Ibrahim: slides 6–10 (training, evaluation, results, demo, errors, conclusion).
- Swap if you prefer. Both of you should know the Q&A section.

**Before the presentation:**
1. About 5 min early: kaggle.com → `sqlbuddy-04-demo` → **Edit** → GPU T4 + Internet on → **Run all** → open the `PUBLIC LINK` (90 min).
   Backup: `python app/app.py` on a laptop.
2. Open the deck in **Present** mode.

Every number below comes from `results/`. If a TA asks "where is that number from?", the answer is the file named in brackets.

---

## Slide 1: Cover ("SQLBuddy", ~30 s)

**On the slide:** the title, "Fine-tuning a small language model to turn English questions into SQL", and the team.

**What to say:**
> "Assalam-o-Alaikum. We are Fatima Malik and Mohammad Ibrahim Salman, and our project is **SQLBuddy**.
>
> SQLBuddy lets someone ask a database a question in plain English, and a small language model that we
> fine-tuned writes the SQL query that answers it.
>
> We'll cover the problem, our tools, dataset and pipeline, how we fine-tuned and evaluated the model, the
> results with a live demo, and finally what we learned and what comes next."

**Likely question:** *"What does SQLBuddy actually do, in one line?"*
→ "You type a question; the model writes SQL; the app runs it read-only and shows the answer table."

---

## Slide 2: Project overview ("From an English question to a working SQL query", ~95 s)

**On the slide:** three cards (Problem, Objectives, Proposed solution) and a real example: the question "Which
teacher teaches the most courses?" → the SQL our fine-tuned model wrote.

**What to say:**
> "**The problem:** a lot of useful data sits in databases, but to
> get it out you have to write SQL, and most people can't. Big models like GPT can translate questions to SQL,
> but every query costs money, and you have to send your database structure to a third party.
>
> In NLP terms this task is called **text-to-SQL**, a kind of **semantic parsing**: we map a natural-language
> question plus a database schema to a formal query that can be executed.
>
> It is harder than it looks, for three reasons:
> 1. The model has to understand the question.
> 2. It has to **link the words to the right tables and columns**. When someone says 'teacher', which table and
>    which column is that?
> 3. It has to work on **databases it has never seen before**, and still produce SQL that actually runs.
>
> **Our objectives** were three: fine-tune a *small open* model that can run on a laptop; prove with a held-out
> test set that fine-tuning beats the same model without fine-tuning; and build a working demo.
>
> **Our solution:** LoRA fine-tuning of Qwen2.5-Coder 1.5B on the Spider dataset, compared with the untouched
> base model.
>
> At the bottom is a real example from our own university database. The question 'Which teacher teaches the
> most courses?' becomes this SQL: it joins teachers with courses, groups by teacher, sorts by the count and
> takes the top one. That is exactly the query our fine-tuned model wrote."

**Likely questions:**
- *"Why not just use ChatGPT?"* → "Cost per query, privacy (the schema leaves your machine) and no control
  over the model. Our model is free, runs locally, and we can measure and improve it ourselves."
- *"What is semantic parsing?"* → "Converting a sentence into a formal, executable meaning representation;
  here the representation is SQL."
- *"Why a small model?"* → "It runs on a laptop and trains on a free GPU. The question we answer is whether
  cheap fine-tuning can make a small model good enough."

---

## Slide 3: Tools and technologies ("Open-source tools on one free GPU", ~75 s)

**On the slide:** three columns of tools, and a row of NLP techniques.

**What to say:**
> "Everything is open source and runs on a free GPU. The tools fall into three groups.
>
> **Model and training:**
> - The base model is **Qwen2.5-Coder 1.5B Instruct**, a pretrained model for code. It already knows SQL
>   syntax, which is why we chose it.
> - **Hugging Face Transformers** loads it and generates the SQL, with PyTorch underneath.
> - **PEFT** adds the LoRA adapter, and **TRL** runs the supervised fine-tuning.
>
> **Data and evaluation:** the data is **Spider**. **SQLite** executes every query the model writes, so we can
> check the answer rather than just the text. The **official Spider evaluation script** gives the exact-match
> metric and the difficulty levels. We use SciPy for the
> significance test.
>
> **Compute and demo:**
> - Training and evaluation ran on a free **Kaggle T4 GPU**.
> - The trained adapter is published on **Hugging Face Hub**, where anyone can download it.
> - The web demo is built with **Gradio**.
>
> **The NLP techniques we used**, on the bottom row:
> - semantic parsing;
> - transfer learning from a pretrained model;
> - prompt engineering, where we write the database schema as text;
> - parameter-efficient fine-tuning with LoRA;
> - a completion-only loss, meaning we only train on the SQL answer;
> - greedy decoding."

**Likely questions:**
- *"Why Qwen2.5-Coder and not Llama or a general chat model?"* → "It was pretrained on code including SQL,
  so it already knows the syntax. The 1.5B size fits a free T4 for training and a laptop for inference.
  A general model would first have to learn SQL; a 7B model would not fit our free GPU for training."
- *"What is TRL / PEFT?"* → "Hugging Face libraries. PEFT implements LoRA; TRL's SFTTrainer does supervised
  fine-tuning on prompt → completion pairs."
- *"Why Kaggle?"* → "A free T4 GPU with 16 GB, about 30 hours a week. Enough for our 2-hour training run."

---

## Slide 4: Dataset ("Spider dataset, split by database", ~2 min 15 s)

**On the slide:** the split table (train 6,560 / validation 352 / test 1,034 / extra test 30) and four
preparation steps.

**What to say:**
> "We use **Spider**, the standard cross-domain text-to-SQL benchmark from Yale. It has about 10,000 questions
> written by students over **200 databases in 138 domains** (concerts, cars, flights, pets, world geography and
> so on). 'Cross-domain' means the model can't memorise one database; it has to learn to read *any* schema.
>
> Every example is an English question, the correct SQL query, and the database it belongs to. The databases
> are real SQLite files with data, so we can actually run the queries.
>
> **Let me show one real example.** The question is: *'What are the distinct creation years of the departments
> managed by a secretary born in state Alabama?'* It belongs to a database with three tables: department, head
> and management. Management links the other two with foreign keys. The correct SQL joins all three tables and
> filters on the state Alabama. Run on the real database, it returns one answer: **1903**. So even a short
> question can need a three-table join, and that is what makes Spider hard.
>
> **Our split, in the table:**
> - We **train** on Spider's training set: 132 databases, 6,560 questions after preparation.
> - We **test** on Spider's dev set: **1,034 questions on 20 databases that never appear in training**. We
>   checked: zero databases are shared.
> - For **validation** we held out **8 whole databases** instead of random questions, so validation is like
>   the test: unseen databases.
> - As an **extra test** we wrote 30 questions for our own university database.
>
> **Preparation, the four steps at the bottom:**
> 1. We read each database's structure from SQLite and write it as **CREATE TABLE statements with primary
>    and foreign keys**. This is what the model sees.
> 2. We make the validation databases **disjoint** from training.
> 3. We **normalise the gold SQL**: Spider's queries have messy spacing like 'a ,  b'.
> 4. We **drop the 88 longest examples**, over 1,536 tokens, instead of cutting them, because cutting would
>    remove the SQL answer.
>
> The test set has four official difficulty levels, based on how complex the correct SQL is: easy 248,
> medium 446, hard 174 and extra hard 166."

**Show one record if asked ("show me the dataset"):**
- **Question:** *"What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"*
- **Database:** `department_management`. Its tables are `department`, `head` and `management`. `management` links the other two through foreign keys.
- **Gold SQL:** `SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id = T2.department_id JOIN head AS T3 ON T2.head_id = T3.head_id WHERE T3.born_state = 'Alabama'`
- **Result:** running it on the real database gives `1903`.

**What a record contains:**
- `db_id`, `question`, `query`: these are what we use.
- `question_toks` and `query_toks`: word-split versions, not used by us.
- `sql`: a parsed tree, which the official evaluation script uses for exact match and difficulty.

**Database files:**
- `tables.json` describes every schema: table and column names, types, primary keys and foreign keys.
- `database/<name>/<name>.sqlite` holds the actual data.

**Numbers to remember:**
- On average a database has 5.3 tables, 27 columns and about 5 foreign keys; the largest has 26 tables and 352 columns.
- The test databases are smaller: 4 tables and 22 columns on average.
- Questions average 12.7 words.

**Likely questions:**
- *"Why test on dev, not Spider's test set?"* → "Spider's official test set was hidden for years: only the
  organisers could score it. Everyone reports on dev, so it is the standard choice."
- *"Why hold out whole databases for validation?"* → "Because the test databases are unseen. Random questions
  would come from databases the model trains on, and the validation score would be too optimistic."
- *"Why 6,560 and not 7,000?"* → "352 went to validation; 88 were longer than 1,536 tokens and were dropped
  rather than truncated."
- *"Why didn't you use train_others.json?"* → "Those 1,659 questions come from older datasets with a
  different style. They would add about 24 % training time; we kept the standard Spider training set."
- *"Where did you get the data?"* → "The questions and SQL come from Hugging Face `xlangai/spider`, which has
  the corrected dev set with 44 fixed gold queries. The SQLite files come from the Kaggle copy of the
  original release."
- *"What are the difficulty levels based on?"* → "The official Spider rule counts SQL components, such as
  JOINs, GROUP BY, nesting and set operations, in the *gold* query."

---

## Slide 5: Methodology and implementation ("System pipeline", ~100 s)

**On the slide:** the pipeline diagram (question + schema → prompt → two branches, baseline and baseline + LoRA
→ extract SQL, run on SQLite → score) and four numbered steps.

**What to say:**
> "Here is the whole system, left to right.
>
> **Step 1:** we read the database structure and write it as CREATE TABLE text with keys, and add the question.
> In front of it there is a short instruction: 'You are an expert SQLite assistant; return only the SQL
> query.' This prompt is **identical for both models**.
>
> **Step 2:** the pipeline splits. The top branch is the **baseline**, Qwen2.5-Coder 1.5B exactly as
> downloaded. The bottom branch is the **same model with our LoRA adapter attached**. So the adapter is the
> *only* difference between the two, which makes it a controlled experiment: any change in the results comes
> from fine-tuning.
>
> **Step 3:** we generate with **greedy decoding**: the model always picks the most likely next token, so
> the output is deterministic and repeatable. We keep the first complete
> SQL query.
>
> **Step 4:** we **run the query on SQLite** and compare the result rows with the correct answer.
>
> **Following one question through:** for 'How many singers do we have?', the prompt holds the concert
> database's four tables plus the question. Both models write `SELECT count(*) FROM singer`. SQLite returns 6,
> the correct query also returns 6, so both models get this one right. For harder questions with joins, the
> two branches start to differ, and that's what the results measure.
>
> Everything is Python with Hugging Face. Training, evaluation and the demo **share one code path**, so the
> demo behaves exactly like the model we evaluated. We checked all of it with 56 unit tests on CPU before
> spending any GPU time."

**The exact prompt (if asked):**
- **System:** "You are an expert SQLite assistant. Given a database schema and a question, write one SQLite
  SELECT query that answers the question. Return only the SQL query, with no explanation."
- **User:** "Database schema:", then the CREATE TABLE statements, then "Question: …".

**Likely questions:**
- *"Why write the schema as CREATE TABLE text?"* → "Code models have seen millions of CREATE TABLE
  statements, so they read that format naturally. It also shows primary and foreign keys, which tell the model
  how tables join. Spider's original schema text is inconsistent, so we rebuild it in one clean format."
- *"Why greedy and not sampling or beam search?"* → "Greedy is deterministic, so results are reproducible and
  the comparison is fair. Sampling would give different answers each run. Beam search costs several times more
  compute; it's a possible extension."
- *"What does 'extract SQL' do?"* → "It takes the first complete SQL statement from the model's output. It
  removes code fences, a leading 'SQL:' and anything after the first semicolon, then cleans whitespace.
  Honestly, on our test set it never changed a score: neither model added extra text. It is a safety net."
- *"Why one shared code path?"* → "So the demo, the training and the evaluation can never disagree, for
  example by using a different prompt."

---

## Slide 6: Fine-tuning ("LoRA: train 1.2 % of the weights", ~100 s)

**On the slide:** the LoRA diagram (input → frozen original model + small trained adapter → added → SQL),
"What we trained", and the loss curve.

**What to say:**
> "Now the fine-tuning itself. Read the diagram left to right. The input goes through the **original model**,
> 1.56 billion weights, which are **frozen**: we never change them. In parallel it goes through a small
> **LoRA adapter**: 18.5 million weights, the only part we train. The two outputs are added together.
>
> Mathematically, for a weight matrix W, LoRA learns a low-rank update B·A, where A and B are thin matrices of
> rank 16, and uses W + B·A. We put these adapters on the attention and feed-forward layers.
>
> That is why LoRA is cheap:
> - only **1.2 % of the weights** are trained;
> - it fits on a free T4 GPU, using 5.4 GB of memory;
> - the adapter file is only **74 MB** instead of 3 GB.
>
> **What we trained:** 6,560 Spider examples for **two epochs**, meaning the model saw every example twice.
> With 16 examples per step, that is 820 optimizer steps. The loss is computed **only on the SQL answer**, not
> on the prompt, so the model learns to write SQL instead of learning to copy schemas. It took **2 hours
> 10 minutes** on the free Kaggle GPU.
>
> **The loss curve on the right:** the blue line, training loss, falls from 0.26 to 0.04. The orange points are validation loss on the 8 unseen databases. It is lowest at step
> 200, 0.184, then rises to 0.22. So there is **some over-fitting in the second epoch**. We fixed all settings in our plan *before* testing, so we evaluated the
> final adapter as planned rather than choosing a checkpoint after seeing test results."

**Settings (if asked):**
- LoRA rank 16, alpha 32, dropout 0.05.
- Adapters on all attention projections (q, k, v, o) and MLP projections (gate, up, down).
- Learning rate 2e-4 with a cosine schedule and 3 % warm-up.
- Batch 4 × 4 gradient accumulation = 16.
- Max length 1,536 tokens.
- **fp16** precision, because the T4 has no bf16. Gradient checkpointing saves memory.

**Likely questions:**
- *"Why LoRA and not full fine-tuning?"* → "Full fine-tuning of 1.5B parameters with the Adam optimizer needs
  about 25 GB of GPU memory; the T4 has 16. LoRA used 5.4 GB, and it reduces the risk of the model forgetting
  what it already knew."
- *"Why rank 16, two epochs, lr 2e-4?"* → "Standard, stable LoRA defaults for a model this size, decided
  before training. We did not search hyper-parameters, and we never tuned on the test set."
- *"Why loss only on the SQL (completion-only)?"* → "The schema is input, not something to generate. Training
  on it would spend most of the learning on copying table definitions."
- *"Validation loss goes up. Isn't the model getting worse?"* → "Loss measures how *confident* the model is
  per token; accuracy measures whether the final answer is right. It became over-confident on some tokens,
  but test accuracy still rose 16 points. The step-200 checkpoint might be even better; early stopping on
  validation accuracy is future work."
- *"Why fp16?"* → "The T4 GPU doesn't support bf16. fp16 mixed precision halves memory and speeds up training."

---

## Slide 7: Evaluation ("Three metrics, one fair comparison", ~85 s)

**On the slide:** three metric cards and the "fair comparison" chips.

**What to say:**
> "How do we measure success? We use three metrics.
>
> **Execution accuracy** is our primary metric. We run the predicted SQL and the correct SQL and check that
> they return the same rows. Row order only matters if the correct query has ORDER BY.
>
> **Exact match** is the official Spider metric. It parses both queries and compares their parts: SELECT,
> WHERE, GROUP BY and so on.
>
> **Here is why we prefer execution accuracy.** 'How many singers do we have?' can be answered with
> `SELECT count(*) FROM singer` or with `SELECT count(singer_id) FROM singer`. Both are correct and return 6.
> Execution accuracy accepts both; exact match can reject the second one only because it is written
> differently. So exact match also rewards copying the annotators' writing style.
>
> **Valid-SQL rate** is simply how often the generated query runs without an error.
>
> To make the comparison fair, the base and fine-tuned models get the **same prompt, the same greedy decoding
> and the same 1,034 unseen questions**. Only the adapter differs.
>
> Because both models answer the *same* questions, the right statistics are **paired**:
> - **McNemar's test** looks only at the questions where the two models disagree;
> - a **bootstrap confidence interval** tells us how big the gain is, with uncertainty.
>
> We also break every result down by the four difficulty levels."

**Likely questions:**
- *"Why is execution accuracy better than exact match?"* → "`SELECT count(*) FROM singer` and
  `SELECT count(singer_id) FROM singer` are both right, but exact match may fail one. Execution checks the
  answer itself."
- *"Can execution accuracy be wrong?"* → "Yes, in two ways. A wrong query can return the right rows by
  coincidence: we found one such case, test_197. And a correct query can fail on a type difference like
  '3' versus 3. Spider's stricter 'test-suite accuracy' runs each query on several database copies; we name
  that as a limitation."
- *"Why McNemar and not a t-test?"* → "The outcomes are paired yes/no results on the same questions. McNemar
  uses only the discordant pairs (fixed vs broken), which is exactly the right test for that."
- *"What is the bootstrap CI?"* → "We resample the 1,034 questions with replacement 10,000 times and recompute
  the gain each time. The middle 95 % of those gains is the interval: +13.2 to +19.1 points."

---

## Slide 8: Results ("The fine-tuned model wins on every metric", ~80 s)

**On the slide:** the bar chart by difficulty, the metric table, 220 fixed vs 53 broken, McNemar p and the CI.

**What to say:**
> "Here are the results on the 1,034 unseen questions. **Execution accuracy rises from 57.0 to 73.1
> percent**, a gain of 16 points. That is about 167 more questions answered correctly.
>
> The chart shows the gain at **every difficulty level**:
>
> | Level | Base → fine-tuned | Gain |
> |---|---|---|
> | easy | 72 → 88 | +16 |
> | medium | 59 → 75 | +15 |
> | **hard** | **40 → 64** | **+24**, the biggest gain |
> | extra hard | 46 → 56 | +10, the smallest gain |
>
> Hard questions need several joins, and that is exactly where the base model was making mistakes. Extra-hard
> questions need nested queries and set operations, which remain difficult.
>
> **Valid SQL rises from 77 to 92 percent**, so far fewer queries crash. **Exact match more than doubles**,
> from 32 to 68, partly because the model learned Spider's writing style. That's why we trust execution
> accuracy more.
>
> **Is it real or luck?** Question by question, fine-tuning **fixed 220 and broke 53**. If fine-tuning did
> nothing, both directions would be equally likely. McNemar's test gives **p = 2.4 × 10⁻²⁵**, practically
> zero, and the 95 % confidence interval for the gain is **+13 to +19 points**, far above zero. It's also
> broad: **18 of the 20 test databases improve**.
>
> Let us show you this live."

**▶ LIVE DEMO (right after this slide, ~1–2 min). The TA may ask to run one query.**
1. Open the Kaggle `PUBLIC LINK`. Point at the status bar: **Model ready**, **GPU: Tesla T4**, **Fine-tuned
   adapter loaded**.
2. Choose **Compare both** and click the example *"What is the average score in each course? Show the course title."*
3. Explain while it runs:
   > "It generates twice: the base model first, then the fine-tuned one. You can see the SQL appear token by token."
4. What you will see:
   - **Base:** a red **SQL error**, `no such column: T2.title`. This is exactly the wrong-alias mistake from
     the next slide: the base model made it 201 times on Spider.
   - **Fine-tuned:** **Done · 12 rows**, the average score per course.
5. Point at the time and token count shown under each panel.
6. If asked about safety, explain it rather than demo it (the model nearly always writes a SELECT anyway):
   > "Every query passes four checks. The database is opened read-only. Only a single SELECT statement is
   > accepted. SQLite's own authorizer blocks any write. And there is a 10-second time limit. Even if the
   > model wrote DELETE or DROP, the app refuses to run it; our unit tests check exactly that
   > (`tests/test_app_safety.py`)."
7. Backup if the internet fails: run the app on the laptop, or show `results/demo_screenshot.png`.

**Likely questions:**
- *"Why is the gain biggest on hard questions?"* → "Hard questions need several JOINs; that's exactly where
  the base model made alias errors, and fine-tuning fixed most of them."
- *"Why only +10 on extra hard?"* → "Those need nested queries and set operations (INTERSECT, EXCEPT); logic
  errors there remain. Most of our remaining failures are in these categories."
- *"53 questions got worse. Why?"* → "The fine-tuned model sometimes over-simplifies and drops a join that was
  needed. For example, for 'Which region is Kabul in?' it treated Kabul as a country instead of a city. This
  fits the over-fitting signal in the loss curve."
- *"How long does one question take?"* → "About 1–4 s on the Kaggle GPU and 3–15 s on a laptop CPU. The model
  runs once per generated token, so time grows with the length of the SQL."

---

## Slide 9: Error analysis and generalization ("Fewer broken queries, and it generalizes", ~100 s)

**On the slide:** left: invalid queries 242 → 85, wrong-alias errors 201 → 61, and "still hard: schema
linking". Right: our own university database, 36.7 % → 73.3 %.

**What to say:**
> "Numbers tell you *how much*; error analysis tells you *why*.
>
> **Left: what fine-tuning fixed.** The biggest effect is that the model **stops writing SQL that doesn't
> run**. Queries that fail to execute fall from **242 to 85**.
>
> The most common bug was a **wrong table alias**. The base model joins two tables, calls them T1 and T2, and
> then takes a column from the wrong one. For example, it writes `T2.department_name` when department_name is
> in T1. That is exactly what you just saw in the demo with `T2.title`.
> These errors fell from **201 to 61**.
>
> **What's still hard is schema linking**: choosing the right table, join or column. We read 50 remaining
> failures by hand, and **27** were schema-linking errors. For example, for 'How many car models are produced
> in the USA?', the model filters on `country = 'USA'`, but in that database the country column holds an ID
> number, so it needed one more join to the countries table. A few other errors come from values the model
> never sees, like writing 'French' when the database stores 'France', because the prompt contains only the
> schema, not the data.
>
> **Right: does it generalize beyond Spider?** We wrote **30 new questions** for our own university database,
> with the correct SQL fixed *before* running either model. The fine-tuned model keeps its Spider level,
> **73 percent**. The base model drops to **37 percent**, mainly because of the same alias errors. It fixed 11
> questions and broke none, p = 0.001. It's only 30 questions, so we present it as supporting evidence."

**Likely questions:**
- *"How did you categorise the errors?"* → "Two ways. First, automatically for all failures, by comparing the
  structure of predicted and gold SQL with the sqlglot parser. Second, by hand for a random sample of 50
  failures, stratified by difficulty (`results/error_sample.csv`)."
- *"Why does the base model do worse on your database than on Spider?"* → "It tends to add unnecessary joins
  and then use the wrong alias. Our questions trigger that more often. The fine-tuned model learned to write
  compact queries."
- *"Couldn't you have written easy questions on purpose?"* → "The questions and gold SQL were fixed before
  running either model, and they cover all four official difficulty levels: 10 easy, 7 medium, 7 hard, 6 extra
  hard. But yes, we wrote them ourselves, which is why we call it supporting evidence."
- *"How would you fix schema linking?"* → "Add a few sample cell values per table to the prompt, or a
  retrieval step that selects only the relevant tables. Both are future work."

---

## Slide 10: Conclusion ("Small model, cheap fine-tuning, big gain", ~80 s)

**On the slide:** four numbers (+16.2, 92 %, 18/20, 1.2 %) and three cards (What we learned, Limitations,
Future work).

**What to say:**
> "To conclude, four numbers:
> - execution accuracy up **16 points**, from 57 to 73 percent;
> - **92 percent** of queries now run, up from 77;
> - **18 of 20** unseen databases improve, and the gain is highly significant;
> - all for training **1.2 percent** of the weights in about two GPU hours.
>
> **What we learned:** a small open model can be specialised cheaply; most of the gain comes from no longer
> writing broken SQL; and the gain carries over to a database we wrote ourselves.
>
> **Limitations, honestly:**
> - the model still sometimes picks the wrong table or column;
> - validation loss shows over-fitting in the second epoch;
> - we ran one training seed on one benchmark;
> - Spider is public, so both models may have seen it during pre-training. That keeps the *comparison* fair,
>   but the absolute numbers may be a bit high.
>
> **Future work:** show sample cell values in the prompt, stop training early based on validation accuracy,
> and test on harder benchmarks like BIRD and Spider 2.0.
>
> So the main message of our project: you don't need a huge model or an expensive API to query a database in
> plain English. A small model with cheap, well-evaluated fine-tuning goes a long way.
>
> Thank you. We're happy to take questions."

**Likely questions:**
- *"What would you do with more time?"* → Give the future-work list, plus the extra baselines in the next section.
- *"Is the 73 % good?"* → "For a 1.5B model with greedy decoding and no extra tricks, it's a solid result.
  Much larger systems score higher on Spider, but our point is the *gain* from cheap fine-tuning, which we
  proved on unseen data."

---

## General TA questions (any slide)

### "Why this approach and not another?"
| Choice | Why | Alternative we rejected |
|---|---|---|
| Fine-tuning (LoRA) | teaches the model the task and the dataset's SQL style permanently, at no cost per query | prompting only (see baselines below); full fine-tuning (doesn't fit a free GPU) |
| Qwen2.5-Coder-1.5B | knows SQL already, small, open | general chat model (weaker at SQL); 7B (doesn't fit for training) |
| Spider | the standard cross-domain benchmark, with real databases to execute | WikiSQL (single tables only, too easy); BIRD (harder, bigger; future work) |
| Execution accuracy | checks the answer, not the writing style | exact match only (punishes correct alternatives) |
| Database-disjoint validation | mirrors the unseen-database test | random split (optimistic) |

### "What baselines did you test? Wouldn't something simpler, like RAG or few-shot prompting, do better?"
Be honest. **What we tested:**
- the **same model without fine-tuning** (zero-shot), with the identical prompt and the identical 1,034 questions;
- a per-database breakdown;
- the out-of-domain database.

**What we did not test:** few-shot prompting, RAG and bigger models.

**Suggested answer:**
> "Our baseline is the same model with the same prompt, so the only difference is fine-tuning, which isolates
> its effect. **RAG** for text-to-SQL would mean retrieving similar training questions with their SQL and
> putting them in the prompt as examples. It needs no training, but every prompt gets longer and slower, and it
> still relies on the base model, which made 201 wrong-alias errors. Fine-tuning removed most of those. We
> can't claim RAG would be worse without testing it, so few-shot / RAG prompting is our first future baseline.
> It would show how much of our gain *only* training can give."

### "Which parts of your pipeline improved performance or speed?"
| Component | Effect | Measured? |
|---|---|---|
| LoRA fine-tuning | execution accuracy 57.0 → 73.1 %, valid SQL 76.6 → 91.8 % | ✅ main result |
| Schema with primary/foreign keys | helps the model find join paths | ❌ no separate ablation (limitation) |
| SQL extraction step | **no effect on scores**: neither model added extra text; scores identical without it | ✅ checked |
| Completion-only loss | standard practice; training focuses on the SQL | ❌ no ablation |
| Prompt cache (schema part computed once per database) | first token after **0.6 s instead of 2.3 s** in the app | ✅ |
| Batched generation, prompts sorted by length | 1,034 test questions generated in under 4 minutes on a T4 | ✅ |
| Model loaded at app start-up | the first question doesn't wait for loading | ✅ |
| Prompt-lookup decoding (a speed trick) | **50 % slower** on CPU, so rejected | ✅ |

Honest line:
> "We measured fine-tuning and our speed changes. We did not run a separate ablation for every prompt detail,
> such as keys in the schema; that's a limitation."

### "Is it running? Run one query."
Follow the live-demo steps after slide 8. If asked **"why read-only?"**:
> "The task and the training data are questions (SELECT) only, the model is wrong about one time in four, and a
> wrong DELETE could destroy data. Real text-to-SQL tools give the model read-only access."

### "Could the model have seen Spider before?"
> "Possibly: Spider is public, and the base model already writes Spider-style SQL. But it affects both models
> equally, so the *difference*, the effect of fine-tuning, is still fair."

### "Why is the demo slow on a laptop?"
> "The model runs once per generated token, about 0.16 s per token on a 4-core CPU, and that time scales with
> the number of cores. It is a hardware limit, not a bug: generation stops at the end of the SQL. On the Kaggle
> GPU it takes 1–4 s. (`docs/PERFORMANCE.md`)"

---

## Number cheat-sheet
| | Base | Fine-tuned |
|---|---|---|
| Execution accuracy | 57.0 % | **73.1 %** |
| Exact match | 32.4 % | **68.0 %** |
| Valid SQL | 76.6 % | **91.8 %** |
| Easy / medium / hard / extra | 72.2 / 59.2 / 39.7 / 46.4 | 88.3 / 74.7 / 63.8 / 56.0 |
| Own database (30 questions) | 36.7 % | 73.3 % |

**Significance:** 220 fixed, 53 broken; McNemar p = 2.4 × 10⁻²⁵; 95 % CI +13.2 to +19.1 points;
18 of 20 databases improve.

**Training:** 18.5 M trainable parameters (1.2 % of 1.56 B), rank 16, 2 epochs, 820 steps,
2 h 10 min on a T4, 5.4 GB GPU memory. Train loss 0.26 → 0.04; validation loss best at step 200
(0.184), final 0.223.

**Data:**
- Train 6,560 (132 DBs); validation 352 (8 DBs); test 1,034 (20 DBs); 0 DBs shared between train and test.
- Test difficulty: 248 / 446 / 174 / 166.

**Errors:**
- Invalid SQL 242 → 85; wrong-alias errors 201 → 61.
- 27 of 50 hand-checked failures are schema linking.

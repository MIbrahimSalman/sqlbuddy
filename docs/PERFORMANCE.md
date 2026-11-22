# Why does a query take several seconds on a laptop? (latency investigation)

**Question:** after the first start, every question still takes noticeable time. Is this a code problem
or the CPU?

**Short answer:** it is mainly the **CPU**. The model has 1.5 billion parameters and must run **once per
generated token**. A typical SQL query is 10–60 tokens, so on a 4-core CPU it takes about 2–15 s. The code
was not wasting work: generation always stopped by itself, at the end of the SQL. We still made four safe
improvements (below), and none of them changes the model's answers.

All measurements were taken on a 4-core Intel Xeon (2.3 GHz) cloud VM with fp32 weights, using the sample
`university.db` and greedy decoding (identical to the evaluation). Times vary by about ±15 % between
identical runs on this VM. Laptops differ, but the *shape* is the same.

## 1. Where the time goes

| Phase | What happens | Measured |
|---|---|---|
| First start: download | ~3 GB model from Hugging Face, once | depends on internet |
| First start: load from disk (cold) | read + convert the weights to fp32 | **118 s** (first read from disk) |
| Later starts: load (warm disk cache) | same, files already cached by the OS | **2–6 s** |
| Per question: read the prompt (*prefill*) | system prompt + schema + question ≈ 280 tokens in one pass | **≈ 2 s** |
| Per question: generate (*decode*) | one full pass of the network **per output token** | **≈ 0.16–0.25 s per token** |

Per-question time ≈ prefill + tokens × time per token:

| Question (sample DB) | output tokens | total time |
|---|---|---|
| How many students are there? | 6 | ≈ 3 s |
| List the names of students from Lahore. (fine-tuned) | 13 | ≈ 5–6 s |
| Which teacher teaches the most courses? | 39–43 | ≈ 9–11 s |
| What is the average score in each course? … | 53–56 | ≈ 11–14 s |

**Compare both** runs the model twice, so it takes about twice as long.

## 2. Evidence that it is hardware-bound, not a code bug

1. **No wasted tokens.** With `max_new_tokens = 256`, the model always stopped on its own, 6–110 tokens
   in, exactly at the end of the SQL. Nothing generates useless text up to the limit.
2. **Time per token scales with CPU cores.** Same single decode step, different thread counts:

   | threads | 1 | 2 | 4 |
   |---|---|---|---|
   | fp32, ms per token | 561 | 255 | 163 |
   | bf16, ms per token | 425 | 226 | 136 |

   Doubling the cores almost halves the time, which is the signature of a compute-bound workload. A laptop
   with more or faster cores is proportionally faster. On a Kaggle T4 GPU, the 53-token "average score"
   query took 4.1 s instead of ≈ 12 s on this CPU.
3. **The work per token is fixed by the model size.** Every token multiplies through 28 layers and a
   1536 × 151,936 output layer, about 3 GFLOP per token. The LoRA adapter adds only ~1 % on top.

## 3. What we changed (identical output, verified)

| Change | Effect | Check |
|---|---|---|
| **Load the model in the background at app start**, with a visible "Loading model…" state | the first question no longer waits for loading | UI test |
| **Prompt-prefix cache**: the system prompt + schema are the same for every question on a database, so their attention keys/values are computed once and reused (separately for base and fine-tuned, because LoRA changes them) | time to first token **≈ 2.3 s → ≈ 0.6 s** for every question after the first one on a DB; the sample DB is warmed up at start | generated tokens identical with and without the cache (8/8 runs) |
| **Token streaming**: the SQL appears while it is being generated | waiting feels shorter; the user sees progress | UI test |
| **Stop button**: generation ends after the current token | no need to wait for an unwanted long answer | test: stops after a few tokens |
| **Apple-silicon GPU (MPS)** is used automatically on Macs; CUDA GPUs are used if present | large speed-up on M1/M2/M3 Macs (not measurable here) | falls back to CPU automatically |

## 4. What we tried and rejected

| Idea | Result | Why rejected |
|---|---|---|
| bf16 weights on CPU | ~15 % faster decode, half the RAM, same SQL on our tests | only fast on CPUs with native bf16 (AVX512-BF16/AMX); on most laptops it is *slower*. Kept as an opt-in: `SQLBUDDY_DTYPE=bf16` |
| Prompt-lookup decoding (guess the next tokens by copying from the schema, verify in one pass; identical greedy output) | **≈ 50 % slower** on CPU (101 s → 154 s over 10 generations) | checking 10 guessed tokens costs more than it saves on a CPU |
| int8 quantisation | would be ~2× faster | changes the model's numbers, so the evaluation results (73.1 %) would no longer be guaranteed for the demo |
| A smaller model (0.5 B) | ~3× faster | a different model from the one we evaluated |

## 5. Practical tips for the presentation

- Start the app a few minutes early: `python app/app.py`. Loading and warm-up then finish before the
  audience arrives, and the status bar shows **Model ready**.
- Use short questions for the live comparison (counts and simple lists: 3–6 s per model).
- Close other heavy programs. Plug the laptop in: on battery, CPUs throttle.
- For a faster demo, use the Kaggle GPU notebook `04_demo`: about 4 s for the longest example question, and a public link.
- Settings: `SQLBUDDY_DEVICE=cpu|cuda|mps`, `SQLBUDDY_DTYPE=fp32|bf16|fp16`.

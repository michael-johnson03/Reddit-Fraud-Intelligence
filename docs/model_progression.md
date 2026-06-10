# Model progression — Flan-T5 → BART-CNN → Mistral-7B

The synthesis model evolved across the 6 production runs. The README has a brief mention; this is the longer story for anyone interested in why each switch happened.

## Run-by-run

| Run | Model | Why |
|---|---|---|
| 2026-03-22 (R1) | `google/flan-t5-large` | Default. Cheapest fast option. Sequence-to-sequence, lightweight, runs on smaller GPUs. |
| 2026-03-29 (R2) | `google/flan-t5-large` | Continuing R1 baseline. Validating cross-run dedup before changing the model. |
| 2026-04-06 (R3) | `google/flan-t5-large` | Final Flan-T5 run. Output quality plateau identified. |
| 2026-04-07 (R4) | `facebook/bart-large-cnn` | Experiment. BART-CNN is summarization-tuned and might handle the narrative better. |
| 2026-04-08 (R5) | `mistralai/Mistral-7B-Instruct-v0.2` (4-bit NF4) | Production. Best output quality. Quantization required to fit on A100 with embedding model and FAISS index. |
| 2026-04-27 (R6) | `mistralai/Mistral-7B-Instruct-v0.2` (4-bit NF4) | Final pre-presentation run. Same setup as R5. |

## Why Flan-T5 didn't work

`flan-t5-large` (~780M parameters) was the cheap-and-fast starting point. Two issues surfaced over R1–R3:

1. **Truncation under retrieval load.** When the retrieved evidence window exceeded ~512 tokens, the model's effective context dropped fast and Pattern output became generic. Flan-T5 supports longer contexts in principle but degraded noticeably on the synthesis quality at our retrieval volumes.

2. **Structured-output drift.** The Pattern / Targets & Impact / Watch signals template held inconsistently. Roughly 30–40% of outputs were missing a section header or had truncated sections. The quality gate caught the worst cases but the LLM-side quality wasn't reliable enough.

Quality assessment by reading the outputs: Flan-T5 syntheses tended to be repetitive ("Multiple users have... multiple users reporting...") and short. The model wanted to extract rather than synthesize.

## Why BART-CNN was worse, not better

`facebook/bart-large-cnn` (~400M parameters) is summarization-tuned on the CNN/DailyMail dataset. The hypothesis: a model specifically trained for summarization might handle "synthesize many short posts into one structured brief" better than a general seq2seq.

The hypothesis failed. BART-CNN is **extractive-leaning** — its training task rewarded staying close to source phrasing. On Reddit posts, that meant:

- The quality gate fired more often than with Flan-T5 because outputs were too close to source post text.
- Even when outputs passed the gate, they read as "selected sentences from posts" rather than "synthesized pattern."
- The structured template held even worse than Flan-T5.

R4 produced one usable run worth of output but the team consensus was to move on.

## Why Mistral-7B worked

Three things changed at R5:

1. **Parameter count** — Mistral-7B is ~10× the parameters of BART-CNN and ~9× Flan-T5-Large. The synthesis-vs-extraction tradeoff is much better in this size class. Instruct-tuned models in particular have stronger "follow this template" behavior.

2. **Instruction tuning** — `Mistral-7B-Instruct-v0.2` uses the `[INST]...[/INST]` prompt format and was explicitly fine-tuned to follow detailed instructions. The structured `Pattern / Targets & Impact / Watch signals` template held reliably across themes.

3. **4-bit NF4 quantization made it feasible** — without quantization, Mistral-7B (fp16) consumes ~14 GB of VRAM. With sentence-transformers (~120 MB) and the FAISS index in the same Colab session, that's tight on A100 40 GB. The 4-bit NF4 quantization (per the QLoRA paper, [Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) drops Mistral to ~4 GB without measurable synthesis-quality loss.

The compute dtype for activations is fp16 — only the weights are quantized. This is the standard pattern for 4-bit LLM inference and is well-supported by `bitsandbytes`.

## What this teaches

Three engineering lessons:

**Bigger isn't always the answer, but for analytical synthesis specifically, it is.** Flan-T5 and BART-CNN are perfectly fine models for shorter tasks (single-sentence classification, abstractive summarization on long single documents). For "read 20 posts and synthesize a pattern across them" — a multi-document grounded-reasoning task — the 7B+instruction-tuned class is the right starting point. We confirmed this empirically rather than assuming it.

**Quantization is a free lunch at our scale.** 4-bit NF4 cuts memory by ~3.5× with minimal quality loss. For deployment on consumer GPUs (A10, L4, even T4 with smaller models), quantization is what makes the analysis affordable. The QLoRA paper is the reference; `bitsandbytes` is the standard library.

**Quality gates need to be model-aware.** The `_is_low_quality()` gate caught more BART-CNN failures than Flan-T5 failures because BART's failure mode was different (extraction vs incoherence). When swapping synthesis models, expect to retune the quality gate thresholds — at minimum, the verbatim-copy detection threshold and the section-presence check.

## Reproducibility

The synthesis model is set in `summarizer/theme_summaries.py`:

```python
SYNTH_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
```

To experiment with other models, change that constant and update the prompt template if switching model family (Llama-3 uses chat-style format; Mistral uses `[INST]...[/INST]`).

The 4-bit NF4 configuration is also in that file:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

`bnb_4bit_use_double_quant=True` adds another small memory saving by quantizing the quantization constants themselves — about 0.4 bits/parameter additional reduction with no measurable quality cost.

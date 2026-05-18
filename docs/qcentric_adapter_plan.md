# Question-Centric Adapter — Design Plan

CLIP-style video encoders are trained for *descriptive* alignment ("this clip looks like this caption"). We want *question* alignment ("this clip contains the information needed to answer this question"). The plan below trains a small adapter on top of frozen InternVideo2 + text-tower features to perform that warp, using Social-IQ oracle timestamps as supervision.

## Existing pieces we build on

- `src/social_memory/encoder.py` — frozen X-CLIP and InternVideo2 dual encoders, both expose `(video, transcript)` → embeddings and a text-only `encode_text`.
- `scripts/modal_internvideo.py` + `docs/internvideo2-modal.md` — InternVideo2 Stage2-1B served on Modal.
- `src/social_memory/data_augmentation/chunking.py::compute_chunks_around_oracle` — already produces chunks plus the `oracle_idx` of the chunk containing the ground-truth segment for each (video, question).
- `src/social_memory/pipelines/evaluate_encoder.py` — zero-shot oracle-finding eval; will be the baseline floor and the harness for evaluating trained adapters.

The supervision the slide's loss requires is therefore already present: c⁺ = oracle chunk, N_hard = sibling chunks from same video, N_easy = chunks from other videos.

## 1. Architecture — two-tower, frozen backbones

**Chunk tower** `f_c(video, transcript) → R^d`
- Frozen InternVideo2 visual feature `v ∈ R^1408`.
- Frozen text-tower feature `t ∈ R^512` for the chunk transcript.
- Adapter: project `v`, `t` separately to a shared bottleneck (256), sum (or concat→MLP), then 2-layer MLP with LayerNorm + GELU.
- Output: L2-normalized `c ∈ R^d`, d = 256 or 512.

**Question tower** `f_q(question) → R^d`
- Same frozen text tower as on the chunk side (weight sharing keeps q and transcript-side t in the same starting basin — adapter only learns the question→description warp).
- 2-layer MLP adapter → L2-norm.

**Why two-tower, not cross-encoder.** Two-tower is indexable for retrieval (encode all chunks once, dot-product at query time). Cross-encoders mix q and c via attention so they score better per pair but force a forward pass per candidate — fine for reranking the top-K later, useless for first-stage retrieval.

## 2. Fusion on chunk side — start simple, ablate up

- **v1**: separate projections then mean → MLP. ~1M params. Ship this first.
- **v2**: cross-attention block (transcript tokens query video tokens). Only if v1 plateaus.
- **Residual gate**: `c = α · adapter(x) + (1 - α) · proj(x)` with learnable α; lets the model fall back to ~CLIP space when training signal is weak. Cheap insurance against overfitting.

## 3. Contrastive head and loss

- Learned temperature τ as `logit_scale = exp(s)`, clamped at `log(100)`, init `s = log(1/0.07)` (CLIP default).
- **Symmetric InfoNCE** (q→c and c→q).
- Denominator matches the slide:
  - hard negs `N_hard(q)` = sibling chunks from same video, weight `λ_hard` (ablate {1, 2, 4}).
  - easy negs `N_easy` = in-batch chunks from other videos.
- **Auxiliary loss (free signal)**: Social-IQ distractor answers are topic-aligned but semantically wrong — perfect text negatives. Add `L_distractor(q, a⁺, a⁻)` using the same text tower so the q-side becomes answer-discriminative, not just chunk-discriminative.
- Total: `L = L_answer + λ_d · L_distractor`, with `λ_d ≈ 0.3`.

## 4. Batch construction (load-bearing detail)

- Sampler: per batch pick **B videos**, take **all chunks of each** (≈ 4–10 chunks). For each video, sample one of its questions → c⁺ is that video's oracle chunk.
- Mask in the loss so that other chunks of the same video go into the hard-neg sum, and only the oracle chunk is positive.
- B ≈ 32, ~6 chunks/video → ~200 candidates in denominator; ~B² easy negs and ~B·(k-1) hard negs per step.

## 5. Data pipeline

- **Pre-compute and cache** frozen InternVideo2 features for every chunk in train + val (one Modal pass). Training then operates on small tensors — fast iteration, no Modal round-trips per epoch.
- **Looser positives**: if the oracle window overlaps multiple chunks, treat all overlapping chunks as positives weighted by IoU. Mitigates coarse oracle annotations.
- **Question paraphrasing**: 1 LLM paraphrase per training question, offline. Cheap augmentation for a small train set.

## 6. Training recipe

- AdamW, lr 1e-4 (adapter), weight decay 0.05.
- bf16, cosine schedule, 5% warmup, 10–20 epochs.
- Backbones strictly frozen. If a follow-up is needed: **LoRA on the last block of the text tower only**.
- Eval per epoch on val: R@1 / R@5 / MRR for oracle-chunk retrieval. Tie-break with downstream Social-IQ accuracy when the retrieved chunk is used as the LLM's context.

## 7. Ablation ladder (in order)

1. Zero-shot frozen InternVideo2 dual encoder (existing `evaluate_encoder.py`) — floor.
2. Linear-only adapter (1 layer) — sanity that *any* warp helps.
3. v1 MLP adapter, easy negs only.
4. + hard negs (full slide loss).
5. + transcript on chunk side.
6. + distractor auxiliary loss.
7. + paraphrase augmentation.
8. v2 cross-attention fusion / LoRA on text tower.

## 8. Concrete file plan

- `src/social_memory/adapters/qcentric.py` — `QuestionAdapter`, `ChunkAdapter`, `QCentricModel.forward → (q, c, logit_scale)`.
- `src/social_memory/training/contrastive.py` — InfoNCE with hard-neg masking, batch sampler, distractor head.
- `src/social_memory/datasets/qcentric.py` — yields `(q, c⁺_id, video_id, [c_sibling_ids], [distractors])`, reads cached features.
- `scripts/precompute_features.py` — one-shot Modal pass; writes `.npz` / parquet per video.
- `scripts/train_qcentric.py` + `configs/qcentric_v1.yaml`.
- Extend `scripts/run_encoder_eval.py` with `--adapter-ckpt` to evaluate trained adapter.

## Risks and tradeoffs

- **Data scale.** Social-IQ train is small. Hedges: tiny adapter, shared text encoder, distractor aux loss, paraphrase augmentation. If we still overfit, the right move is *more synthetic questions per chunk* via an LLM, not a bigger adapter.
- **Oracle granularity.** Oracle windows can be loose. IoU-weighted positives prevent the model from being punished for "almost-right" chunks.
- **Two-tower ceiling.** Compositional queries ("the moment *after* X says Y") are hard for any two-tower model. When that ceiling is hit, add a cross-encoder reranker over top-K from the dual encoder — don't make the towers fancier.

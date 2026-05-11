# Hosting InternVideo2 on Modal — Field Notes

Reference for `scripts/modal_internvideo.py` (currently runs **InternVideo2-Stage2-1B**, served on Modal, queried from local Python via `RemoteInternVideoEncoder` in `src/social_memory/encoder.py`).

## Why Stage2-1B, not CLIP-6B / CLIP-1B

InternVideo2 ships in three families. They're easy to confuse because all reuse the same vision tower.

| Variant | Vision params | Text encoder | Weight footprint | Loader complexity |
|---|---|---|---|---|
| **Stage2-1B** ✅ used here | 1B | BERT-large | ~2.8 GB single file | Demo helper, ungated |
| Stage2-6B | 6B | BERT-large | ~12 GB | Same path, bigger GPU |
| CLIP-1B | 1B | LLaMA-7B + InternVL-13B | ~40 GB across 4 repos | LLaMA-gated, complex loader |
| CLIP-6B | 6B | LLaMA-7B + InternVL-13B | ~50 GB across 4 repos | LLaMA-gated, complex loader |
| CLIP-S14/B14/L14 | ~50M–300M | small CLIP-style | ~1 GB | Standalone, distilled |

The CLIP variants are NOT a small upgrade — even the "1B" one drags in a ~13B-param LLaMA text stack and four separate HF repos (one of them LLaMA-licensed). Per OpenGVLab's own MSRVTT T2V benchmark, **Stage2-1B (51.9) beats CLIP-1B (50.0) and approaches CLIP-6B (50.9)**, so for our use case Stage2-1B is both simpler and stronger.

The 1B/6B labels refer only to the vision tower; the text side is the same. Same vision architecture appears in Stage1, Stage2, and CLIP variants — only the text encoder and alignment head swap.

## Vision encoder architecture (shared across variants)

Custom 3D ViT, EVA/LLaMA-flavored:

- Spatial patches 14×14 at 224 input → 16×16 grid
- `tubelet_size=1` (each of 8 frames becomes its own time row of tokens; no temporal patching)
- Joint spatiotemporal self-attention, not factorized
- QK-normalization, fused RMSNorm, fused MLP, flash-attn, LayerScale
- Attention-pool head → `clip_embed_dim=768` projection

| Variant | Depth | Width | Heads | ~Params |
|---|---|---|---|---|
| 1B | 40 | 1408 | 16 | ~1B |
| 6B | 48 | 3200 | 25 | ~6B |

Stage2-1B was pretrained via masked video modeling + CLIP-teacher distillation (Stage1), then multimodal contrastive alignment to BERT-large (Stage2).

## Modal infrastructure choices

- **Base image**: `nvidia/cuda:12.3.2-devel-ubuntu22.04`. Picked 12.3 specifically because flash-attn publishes prebuilt wheels for `cu123 + torch 2.4 + cp310` but not `cu121` — see "flash-attn wheel" below.
- **GPU**: A10G (24 GB). 1B fp32 model + activations fits comfortably.
- **Volume** `internvideo2-weights` mounted at `/weights`: holds the 2.8 GB Stage2 checkpoint + the 1.3 GB BERT-large dir. Persists across runs so cold start doesn't re-download.
- **Secret** `huggingface` with `HF_TOKEN`: needed because Stage2-1B is a gated repo on HF.
- **scaledown_window=600**: containers stay warm 10 min between calls — encodes inside that window are ~400 ms each instead of paying the ~60 s cold start.
- **Mode**: `modal run` (one-shot ephemeral) for testing; `modal deploy` + `modal.Cls.from_name(...)` for persistent use from local Python. Our `RemoteInternVideoEncoder` wrapper hides the lookup.

Data flow is just RPC: local samples frames with PyAV, ships a numpy array + text via cloudpickle, server returns `{video, text, fused}` numpy embeddings. ~600 KB/clip if you pre-resize to 224×224; up to ~25 MB/clip for raw 1080p frames.

## Gotchas (in order encountered)

### 1. HF token must allow gated repos

Modal secret key must be exactly `HF_TOKEN`. The token itself must either be a classic Read token, or a fine-grained token with **"Read access to contents of all public gated repos you can access"** enabled. A token without that scope returns 403 with "Please enable access to public gated repositories in your fine-grained token settings" — which is distinct from the "not in the authorized list" 403 you get if you haven't accepted the license on the model page yet.

Both gates have to be cleared: (a) accept the license on https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 while logged in as the token owner, AND (b) make sure the token can read gated repos.

### 2. flash-attn wheel availability dictates CUDA version

flash-attn publishes prebuilt wheels with names like `flash_attn-{ver}+cu{X}torch{Y}cxx11abi{Z}-cp{PY}-...whl`. For torch 2.4 + cp310 the matrix is:

| flash-attn version | Available CUDA tags |
|---|---|
| 2.5.8 | cu118, cu122 (no cu121, no cu123) |
| 2.5.9.post1 | cu118, cu122, cu123 |
| 2.6.x | cu118, cu123 |

Pinning `flash-attn==2.5.8` (the original guess) makes its `setup.py` build, which probes `https://...cu122torch2.4...whl` (404 in some asset releases), falls back to source compile, then fails because the CUDA-devel image doesn't ship `clang++`. The fix is to **install a known-good wheel directly by URL** rather than letting setup.py guess:

```python
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)
```

CUDA 12.3 base + torch 2.4.0+cu121 (which is what pytorch publishes) works fine — torch 2.4 cu121 binaries run against a cu123 toolkit without issue.

### 3. flash-attn wheel omits `dropout_layer_norm`

The prebuilt flash-attn wheel does NOT include the optional `dropout_layer_norm` CUDA extension (it lives in `csrc/layer_norm` and must be compiled separately, ~10 min). InternVideo2's `internvl_clip_vision.py` does an unconditional top-level `from flash_attn.ops.rms_norm import DropoutAddRMSNorm`, which transitively `import dropout_layer_norm`s.

Since we run Stage2 in fp32 (no flash-attn fused ops actually invoked), an **empty stub module** satisfies the import without needing to compile anything. Verified by inspecting `flash_attn/ops/layer_norm.py`: every reference to `dropout_layer_norm` is inside function bodies, never module-level — so the import succeeds and class definitions stand; the symbols would only fail if called.

```bash
echo 'def __getattr__(name):' > /usr/local/lib/python3.10/site-packages/dropout_layer_norm.py
echo '    raise RuntimeError("stub")' >> /usr/local/lib/python3.10/site-packages/dropout_layer_norm.py
```

### 4. `peft` version vs `transformers` version

Latest `peft` requires `transformers>=4.43` (for `EncoderDecoderCache`). Pinning `transformers==4.40` then leaving peft unpinned fails. The repo also doesn't actually use the LoRA paths peft enables — peft only gets imported because `models/backbones/internvideo2/__init__.py` eagerly loads `internvideo2_clip_text.py` (the CLIP-6B LLaMA text encoder, which we don't use).

Pinning `peft==0.11.1` is compatible with transformers 4.40, but the better fix is the next gotcha.

### 5. Repo `__init__.py` files eagerly load every variant

`InternVideo2/multi_modality/models/__init__.py` is:

```python
from .internvideo2_clip import InternVideo2_CLIP
from .internvideo2_clip_small import InternVideo2_CLIP_small
from .internvideo2_stage2_visual import InternVideo2_Stage2_visual
from .internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual
```

These run on any `from models.X import Y` — so just importing `setup_internvideo2` from `demo/utils.py` drags in:

- CLIP-1B / CLIP-6B vision + LLaMA-7B text encoders (need `dropout_layer_norm`, `peft`, etc.)
- Stage2 audiovisual variant (needs audio deps we don't have)
- `models/criterions.py` (has broken relative imports — next gotcha)

**Fix:** empty both `models/__init__.py` and `models/backbones/internvideo2/__init__.py` during image build, then import the one function we need (`pretrain_internvideo2_1b_patch14_224`) via its full sub-module path, and drop demo/utils.py's unused `get_sim` import (which is what was pulling in `criterions.py`).

```bash
: > models/__init__.py
: > models/backbones/internvideo2/__init__.py
sed -i 's|from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224|from models.backbones.internvideo2.internvideo2 import pretrain_internvideo2_1b_patch14_224|' demo/utils.py
sed -i 's|from models.criterions import get_sim|# removed|' demo/utils.py
```

### 6. Broken parent-relative imports

`models/criterions.py` uses `from ..utils.distributed import ...` — that only works if `models` is a sub-package of something (i.e. `multi_modality` is itself a package). But `demo/utils.py` does `from models.backbones... import X` as if `models` is top-level. These two import styles can't both be true. The repo gets away with it when run interactively from inside `multi_modality/` because nothing actually exercises the `from ..` path in that flow.

Avoiding the import of `criterions.py` (gotcha #5) sidesteps this entirely.

### 7. `Config.from_file` doesn't resolve `${...}` templates

The demo config uses interpolation syntax inherited from MMEngine-style configs:

```python
text_encoder="${TextEncoders[${text_enc}]}",
num_frames="${num_frames}",
max_txt_l=dict(image="${max_txt_l}", ...),
```

`Config.from_file()` returns the raw values with these strings unresolved. The resolution happens in `eval_dict_leaf`, which is called from `Config.get_config()` but NOT from `from_file()`. Calling it manually fixes `text_encoder`, `num_frames`, `max_txt_l`, and any other unresolved templates in one shot:

```python
from utils.config import Config, eval_dict_leaf
cfg = Config.from_file(...)
cfg = eval_dict_leaf(cfg)
```

This is cleaner than materializing `cfg.model.text_encoder` from `configs/model.py`'s `TextEncoders["bert_large"]` by hand — the manual approach only fixes that one field and leaves the others as literal `"${...}"` strings, which detonate when used as ints.

### 8. fp32 mode required (for now)

The demo config wires `use_flash_attn`, `use_fused_rmsnorm`, `use_fused_mlp` to the value of `use_half_precision` *at file-parse time*. So setting `cfg.use_half_precision = True` after `Config.from_file` doesn't propagate. To run in fp32 (no flash-attn fused ops, no dropout_layer_norm dependency) we set:

```python
cfg.use_half_precision = False
cfg.use_bf16 = False
cfg.model.vision_encoder.use_flash_attn = False
cfg.model.vision_encoder.use_fused_rmsnorm = False
cfg.model.vision_encoder.use_fused_mlp = False
```

fp32 makes encodes ~3-4× slower than fp16 (~400 ms vs ~150 ms) but avoids the source-build detour for the layer_norm extension. To switch to fp16 later: build `flash-attention/csrc/layer_norm` from source in the image (`TORCH_CUDA_ARCH_LIST='8.6'` for A10G; add `9.0` for H100), then drop the `dropout_layer_norm` stub and flip these flags back to `True`.

## Operating the deployment

```bash
# One-time
modal secret create huggingface HF_TOKEN=hf_...                  # gated-repos scope
modal run scripts/modal_internvideo.py::download_weights         # ~3 min, populates volume
modal deploy scripts/modal_internvideo.py                        # for persistent use

# Smoke test (ephemeral)
modal run scripts/modal_internvideo.py --video-path <mp4> --text "..."
```

From local Python (after `modal deploy`):

```python
from social_memory.encoder import RemoteInternVideoEncoder, sample_frames
from pathlib import Path
import numpy as np

enc = RemoteInternVideoEncoder()
frames = sample_frames(Path("clip.mp4"), enc.num_frames)
out = enc(frames, "a person talking")
# out["video_embeddings"], out["text_embeddings"], out["fused_embeddings"]
```

## If anything breaks again

- HF 403 errors: check which 403 (gated-repo vs token-scope) — message is different.
- New `ModuleNotFoundError`: probably another transitive dep from the repo's eager-import chain. Compare against `InternVideo2/multi_modality/requirements.txt`.
- "attempted relative import": something in the import chain hit a `from ..` outside its package. Empty the offending `__init__.py` or use full sub-module paths.
- `'str' object has no attribute X`: an unresolved `${...}` template. Make sure `eval_dict_leaf(cfg)` is called after `Config.from_file`.
- Cold start much longer than 60 s: volume probably didn't mount (check Modal UI Storage tab); model is re-downloading from HF.

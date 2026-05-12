"""Model-agnostic HuggingFace video+text dual encoder."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Curated checkpoint registry
# ---------------------------------------------------------------------------
# Keys are short aliases; values carry the HF hub path and inferred model_type.
# X-CLIP is the only native video+text model in core transformers; CLIP and
# SigLIP are image+text models used here via per-frame encoding + mean pooling.

CHECKPOINT_REGISTRY: dict[str, dict] = {
    # X-CLIP — temporal cross-frame attention, native video understanding
    "xclip-base-16":  {"checkpoint": "microsoft/xclip-base-patch16-16-frames", "model_type": "xclip"},
    "xclip-base-32":  {"checkpoint": "microsoft/xclip-base-patch32",           "model_type": "xclip"},
    # CLIP — strong zero-shot baseline, video via frame averaging
    "clip-vit-b32":   {"checkpoint": "openai/clip-vit-base-patch32",           "model_type": "clip"},
    "clip-vit-l14":   {"checkpoint": "openai/clip-vit-large-patch14",          "model_type": "clip"},
    # SigLIP — sigmoid-loss CLIP variant, generally stronger retrieval
    "siglip-base":      {"checkpoint": "google/siglip-base-patch16-224",         "model_type": "siglip"},
    "siglip-so400m":    {"checkpoint": "google/siglip-so400m-patch14-384",       "model_type": "siglip"},
    # SigLIP2 — improved SigLIP with better scaling and multilingual support
    "siglip2-so400m":   {"checkpoint": "google/siglip2-so400m-patch14-384",      "model_type": "siglip2"},
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class HFVideoTextEncoder(ABC, nn.Module):
    """
    Base class for HuggingFace video+text dual encoders.

    Subclasses implement encode_video / encode_text for a specific model family
    and must set self.num_frames in __init__.

    Built-in model types: "xclip", "clip", "siglip".
    For models requiring trust_remote_code (e.g. LanguageBind, VideoLLaMA):
      1. Subclass HFVideoTextEncoder and implement encode_video / encode_text.
      2. Call HFVideoTextEncoder.register("your_model_type", YourEncoderClass)
         before using from_pretrained.
      3. Pass trust_remote_code=True to from_pretrained.
    """

    _registry: dict[str, type["HFVideoTextEncoder"]] = {}

    num_frames: int

    @classmethod
    def register(cls, model_type: str, encoder_cls: type["HFVideoTextEncoder"]) -> None:
        """Register a custom encoder class for a given HF model_type."""
        cls._registry[model_type] = encoder_cls

    @abstractmethod
    @torch.inference_mode()
    def encode_video(self, frames: list[list[np.ndarray]]) -> torch.Tensor:
        """Encode a batch of video chunks. Returns (n_chunks, embed_size)."""
        ...

    @abstractmethod
    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings. Returns (n_texts, embed_size)."""
        ...

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, trust_remote_code: bool = False, **kwargs
    ) -> "HFVideoTextEncoder":
        """Auto-detect model family from HF config and return the right encoder."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
        model_type = config.model_type

        builtin: dict[str, type[HFVideoTextEncoder]] = {
            "xclip":   XCLIPHFEncoder,
            "clip":    CLIPHFEncoder,
            "siglip":  SiglipHFEncoder,
            "siglip2": Siglip2HFEncoder,
        }
        registry = {**builtin, **cls._registry}

        if model_type not in registry:
            raise ValueError(
                f"No encoder registered for model_type={model_type!r}. "
                f"Built-in: {list(builtin)}. "
                f"Custom registered: {list(cls._registry)}. "
                f"See CHECKPOINT_REGISTRY for known-good checkpoints, or call "
                f"HFVideoTextEncoder.register(model_type, YourEncoderClass) to add one."
            )

        return registry[model_type](checkpoint, trust_remote_code=trust_remote_code, **kwargs)


# ---------------------------------------------------------------------------
# X-CLIP  (native video+text dual encoder)
# ---------------------------------------------------------------------------

class XCLIPHFEncoder(HFVideoTextEncoder):
    """X-CLIP video+text encoder (microsoft/xclip-*)."""

    def __init__(self, checkpoint: str, **_):
        super().__init__()
        from transformers import XCLIPModel, XCLIPProcessor

        self.processor = XCLIPProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = XCLIPModel.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.num_frames: int = self.model.config.vision_config.num_frames

    @torch.inference_mode()
    def encode_video(self, frames: list[list[np.ndarray]]) -> torch.Tensor:
        device = self.model.device
        video_inputs = self.processor.image_processor(frames, return_tensors="pt").to(device)
        out = self.model.get_video_features(pixel_values=video_inputs["pixel_values"])
        return F.normalize(out.pooler_output, dim=-1)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = self.model.device
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        out = self.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        return F.normalize(out.pooler_output, dim=-1)


# ---------------------------------------------------------------------------
# Frame-averaged image+text encoders  (CLIP, SigLIP)
# ---------------------------------------------------------------------------

class _FrameAveragedEncoder(HFVideoTextEncoder):
    """
    Shared base for image-language models used as video encoders via frame
    averaging: encode each frame independently, then mean-pool and re-normalise.

    num_frames has no model-defined value here; DEFAULT_NUM_FRAMES is used
    unless the subclass or caller overrides it.
    """

    DEFAULT_NUM_FRAMES = 32

    def __init__(self, checkpoint: str, num_frames: int = DEFAULT_NUM_FRAMES, **_):
        super().__init__()
        self.model, self.processor = self._load(checkpoint)
        self.model.requires_grad_(False)
        self.num_frames = num_frames

    @staticmethod
    def _load(checkpoint: str):
        raise NotImplementedError

    @torch.inference_mode()
    def encode_video(self, frames: list[list[np.ndarray]]) -> torch.Tensor:
        """Encode each chunk by averaging its per-frame embeddings."""
        device = self.model.device
        chunk_embeds = []
        for chunk_frames in frames:
            inputs = self.processor(images=chunk_frames, return_tensors="pt").to(device)
            frame_embeds = self.model.get_image_features(**inputs)       # (F, D)
            frame_embeds = F.normalize(frame_embeds, dim=-1)
            chunk_embed = F.normalize(frame_embeds.mean(dim=0), dim=-1)  # (D,)
            chunk_embeds.append(chunk_embed)
        return torch.stack(chunk_embeds)                                  # (N, D)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = self.model.device
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_embeds = self.model.get_text_features(**inputs)
        return F.normalize(text_embeds, dim=-1)


class CLIPHFEncoder(_FrameAveragedEncoder):
    """Frame-averaged CLIP encoder (openai/clip-vit-*)."""

    @staticmethod
    def _load(checkpoint: str):
        from transformers import CLIPModel, CLIPProcessor
        return CLIPModel.from_pretrained(checkpoint), CLIPProcessor.from_pretrained(checkpoint)


class SiglipHFEncoder(HFVideoTextEncoder):
    """
    Frame-averaged SigLIP encoder (google/siglip-*).

    Loads SiglipTokenizer and SiglipImageProcessor separately to avoid a
    bug in some transformers versions where SiglipProcessor.from_pretrained
    fails due to a None entry in TOKENIZER_MAPPING_NAMES.
    """

    DEFAULT_NUM_FRAMES = 32

    def __init__(self, checkpoint: str, num_frames: int = DEFAULT_NUM_FRAMES, **_):
        super().__init__()
        from transformers import AutoTokenizer, SiglipImageProcessor, SiglipModel

        self.model = SiglipModel.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.image_processor = SiglipImageProcessor.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.num_frames = num_frames

    @torch.inference_mode()
    def encode_video(self, frames: list[list[np.ndarray]]) -> torch.Tensor:
        device = self.model.device
        chunk_embeds = []
        for chunk_frames in frames:
            inputs = self.image_processor(images=chunk_frames, return_tensors="pt").to(device)
            frame_embeds = self.model.get_image_features(**inputs)       # (F, D)
            frame_embeds = F.normalize(frame_embeds.pooler_output, dim=-1)
            chunk_embed = F.normalize(frame_embeds.mean(dim=0), dim=-1)  # (D,)
            chunk_embeds.append(chunk_embed)
        return torch.stack(chunk_embeds)                                  # (N, D)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = self.model.device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        return F.normalize(self.model.get_text_features(**inputs).pooler_output, dim=-1)


class Siglip2HFEncoder(HFVideoTextEncoder):
    """Frame-averaged SigLIP2 encoder (google/siglip2-*). Video via frame averaging."""

    DEFAULT_NUM_FRAMES = 32

    def __init__(self, checkpoint: str, num_frames: int = DEFAULT_NUM_FRAMES, **_):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.num_frames = num_frames

    @torch.inference_mode()
    def encode_video(self, frames: list[list[np.ndarray]]) -> torch.Tensor:
        device = self.model.device
        chunk_embeds = []
        for chunk_frames in frames:
            inputs = self.processor(images=chunk_frames, return_tensors="pt").to(device)
            frame_embeds = self.model.get_image_features(**inputs)       # (F, D)
            frame_embeds = F.normalize(frame_embeds.pooler_output, dim=-1)
            chunk_embed = F.normalize(frame_embeds.mean(dim=0), dim=-1)  # (D,)
            chunk_embeds.append(chunk_embed)
        return torch.stack(chunk_embeds)                                  # (N, D)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = self.model.device
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        return F.normalize(self.model.get_text_features(**inputs).pooler_output, dim=-1)

"""Custom LangChain ChatModel wrapper for VideoLLaMA2"""
import base64
import tempfile
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class VideoLLaMA2ChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper for VideoLLaMA2.

    Expects HumanMessage content to be a list of content blocks, e.g.:
        HumanMessage(content=[
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What is happening in this video?"},
        ])

    Model string format: videollama2:<model_id>
    e.g. "videollama2:DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
    """

    model_id: str = Field(description="HuggingFace model ID for VideoLLaMA2")
    max_new_tokens: int = 512
    temperature: float = 0.0

    # Loaded lazily on first call
    _model: Any = None
    _processor: Any = None
    _loaded: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._loaded = True

    def _extract_video_and_text(self, message: HumanMessage) -> tuple[Optional[str], str]:
        """Parse a HumanMessage into (video_path, text_prompt).

        Supports two content block formats:
        - {"type": "video", "path": "/path/to/video.mp4"}  (direct path)
        - {"type": "media", "mime_type": "video/mp4", "data": "<base64>"}  (base64-encoded)
        """
        if isinstance(message.content, str):
            return None, message.content

        video_path: Optional[str] = None
        text_parts: List[str] = []
        for block in message.content:
            if isinstance(block, dict):
                if block.get("type") == "video":
                    video_path = block.get("path")
                elif block.get("type") == "media" and "video" in block.get("mime_type", ""):
                    # Decode base64 video data to a temp file
                    video_bytes = base64.b64decode(block["data"])
                    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    tmp.write(video_bytes)
                    tmp.close()
                    video_path = tmp.name
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
        return video_path, " ".join(text_parts)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        self._load()

        # Use the last HumanMessage for video + prompt
        human_msg = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if human_msg is None:
            raise ValueError("No HumanMessage found in messages.")

        video_path, text_prompt = self._extract_video_and_text(human_msg)

        inputs: Dict[str, Any] = self._processor(
            text=text_prompt,
            videos=video_path,
            return_tensors="pt",
        ).to(self._model.device, torch.float16)

        max_new_tokens = kwargs.get("max_tokens", self.max_new_tokens)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated = self._processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=generated))])

    @property
    def _llm_type(self) -> str:
        return "videollama2"

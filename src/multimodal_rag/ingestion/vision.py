from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from multimodal_rag.config import Settings

logger = logging.getLogger(__name__)


class VisionCaptioner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._strict = settings.strict_api_only_mode()
        self._enabled = bool(settings.openai_api_key)
        self._model_name = settings.vision_model
        self._api_key = settings.openai_api_key
        self._llm = None
        if self._enabled:
            try:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=self._model_name,
                    api_key=self._api_key,
                    temperature=0.0,
                )
            except Exception as exc:
                if self._strict:
                    raise RuntimeError("OpenAI vision client is unavailable in strict API-only mode.") from exc
                logger.warning("OpenAI vision client unavailable, using local fallback: %s", exc)
        elif self._strict:
            raise RuntimeError("OpenAI API key is required for vision captioning in strict API-only mode.")

    @staticmethod
    def _to_data_url(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def caption(self, image_path: Path) -> str:
        if not self._llm:
            if self._strict:
                raise RuntimeError("Vision captioning requires OpenAI in strict API-only mode.")
            return f"Image file named {image_path.name}."
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe this image for retrieval in a multimodal RAG system. "
                            "Mention visible objects, labels, numbers, charts, and scene context."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._to_data_url(image_path),
                        },
                    },
                ]
            )
            response = self._llm.invoke(
                [
                    SystemMessage(
                        content="You write concise factual image descriptions for semantic retrieval."
                    ),
                    message,
                ]
            )
            return str(response.content).strip()
        except Exception as exc:  # pragma: no cover - external model/network branch
            if self._strict:
                raise RuntimeError("OpenAI vision captioning failed in strict API-only mode.") from exc
            logger.warning("Vision captioning failed for %s: %s", image_path, exc)
            return f"Image file named {image_path.name}."

    def allow_local_ocr(self) -> bool:
        return not self._strict


def run_ocr(image_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception:  # pragma: no cover - OCR runtime branch
        return ""

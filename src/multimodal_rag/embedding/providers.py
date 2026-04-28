from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from multimodal_rag.config import Settings
from multimodal_rag.embedding.hash_embedder import HashEmbedder
from multimodal_rag.ingestion.vision import VisionCaptioner

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._strict = settings.strict_api_only_mode()
        self._fallback = HashEmbedder(dimensions=384)
        self._openai = None

        if not settings.openai_api_key:
            if self._strict:
                raise RuntimeError("OpenAI API key is required when strict API-only mode is enabled.")
            return

        try:
            from langchain_openai import OpenAIEmbeddings

            self._openai = OpenAIEmbeddings(
                model=settings.text_embedding_model,
                api_key=settings.openai_api_key,
            )
        except Exception as exc:
            if self._strict:
                raise RuntimeError("OpenAI embeddings client is unavailable in strict API-only mode.") from exc
            logger.warning("OpenAI embeddings unavailable, using hash fallback: %s", exc)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._openai:
            try:
                return self._openai.embed_documents(texts)
            except Exception as exc:  # pragma: no cover - network/model branch
                if self._strict:
                    raise RuntimeError("OpenAI embeddings failed in strict API-only mode.") from exc
                logger.warning("OpenAI embeddings failed, using hash fallback: %s", exc)
        return self._fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._openai:
            try:
                return self._openai.embed_query(text)
            except Exception as exc:  # pragma: no cover - network/model branch
                if self._strict:
                    raise RuntimeError("OpenAI query embedding failed in strict API-only mode.") from exc
                logger.warning("OpenAI query embedding failed, using hash fallback: %s", exc)
        return self._fallback.embed_query(text)


class VisionEmbedder:
    """Image-aware embedder with strict OpenAI-only mode support."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._strict = settings.strict_api_only_mode()
        self._fallback = HashEmbedder(dimensions=384)
        self._captioner = VisionCaptioner(settings)
        self._text_embedder = TextEmbedder(settings)
        self._clip = None
        self._clip_attempted = False

    def _get_clip(self):
        if self._strict:
            return None
        if self._clip_attempted:
            return self._clip
        self._clip_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._clip = SentenceTransformer("clip-ViT-B-32")
        except Exception:
            self._clip = None
        return self._clip

    def _openai_caption_embeddings(self, image_paths: list[Path], fallback_texts: list[str]) -> list[list[float]]:
        texts: list[str] = []
        for image_path, fallback_text in zip(image_paths, fallback_texts, strict=False):
            caption = self._captioner.caption(image_path).strip()
            texts.append(caption or fallback_text or image_path.name)
        return self._text_embedder.embed_documents(texts)

    def embed_images(self, image_paths: list[Path], fallback_texts: list[str]) -> list[list[float]]:
        if not image_paths:
            return []

        if self.settings.openai_api_key:
            try:
                return self._openai_caption_embeddings(image_paths, fallback_texts)
            except Exception as exc:  # pragma: no cover - network/model branch
                if self._strict:
                    raise RuntimeError("OpenAI vision embedding failed in strict API-only mode.") from exc
                logger.warning("OpenAI image understanding failed, using local fallback: %s", exc)

        clip = self._get_clip()
        if clip:
            try:
                from PIL import Image

                images = [Image.open(path).convert("RGB") for path in image_paths]
                vectors = clip.encode(images, convert_to_numpy=True, normalize_embeddings=True)
                return vectors.astype(np.float32).tolist()
            except Exception as exc:  # pragma: no cover - model runtime branch
                logger.warning("CLIP image encoding failed, using text fallback: %s", exc)

        if self._strict:
            raise RuntimeError("OpenAI API key is required for image embeddings in strict API-only mode.")
        return self._fallback.embed_documents(fallback_texts)

    def embed_query(self, text: str) -> list[float]:
        if self.settings.openai_api_key:
            try:
                return self._text_embedder.embed_query(text)
            except Exception as exc:  # pragma: no cover - network/model branch
                if self._strict:
                    raise RuntimeError("OpenAI query embedding failed in strict API-only mode.") from exc
                logger.warning("OpenAI query embedding failed, using local fallback: %s", exc)

        clip = self._get_clip()
        if clip:
            try:
                vector = clip.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return vector.astype(np.float32).tolist()
            except Exception as exc:  # pragma: no cover - model runtime branch
                logger.warning("CLIP query encoding failed, using hash fallback: %s", exc)

        if self._strict:
            raise RuntimeError("OpenAI API key is required for query embeddings in strict API-only mode.")
        return self._fallback.embed_query(text)

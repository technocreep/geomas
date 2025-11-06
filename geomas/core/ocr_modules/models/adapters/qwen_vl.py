"""Qwen2.5-VL adapter aligned with the official transcription recipe."""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image

from ..base import BaseOCR, OCRItem, OCRResult
from ...io.cache import get_cached_model
from ...io.hashing import short_hash
from ...io.offline import offline_mode
from ...logging.metrics import observe_latency

SYSTEM_PROMPT = (
    "You are a careful OCR assistant. Transcribe the provided document page "
    "into clean Markdown, preserving structure."
)
USER_PROMPT = "Respond with the Markdown transcription of this page."


@contextmanager
def _open_image(path: Path) -> Iterator[Image.Image]:
    """Return ``path`` as an RGB image, handling PDFs via pdfium when needed."""

    suffix = path.suffix.lower()
    if suffix != ".pdf":
        with Image.open(path) as img:
            converted = img.convert("RGB")
        try:
            yield converted
        finally:
            converted.close()
        return

    pymupdf = None
    for candidate in ("pymupdf", "fitz"):
        try:
            pymupdf = importlib.import_module(candidate)
        except (ModuleNotFoundError, ImportError):
            continue
        else:
            break

    if pymupdf is not None:
        document = pymupdf.open(path)
        try:
            page = document[0]
            colorspace = getattr(pymupdf, "csRGB", None)
            if colorspace is not None:
                pix = page.get_pixmap(colorspace=colorspace, alpha=False)
            else:  # pragma: no cover - exercised when colorspace unavailable
                pix = page.get_pixmap(alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            document.close()
        try:
            yield image
        finally:
            image.close()
        return

    pdfium = importlib.import_module("pypdfium2")
    document = pdfium.PdfDocument(str(path))
    pil_image = None
    try:
        page = document[0]
        pil_image = page.render(scale=1, rotation=0).to_pil()
        converted = pil_image.convert("RGB")
    finally:
        document.close()
        if pil_image is not None:
            pil_close = getattr(pil_image, "close", None)
            if callable(pil_close):
                pil_close()
    try:
        yield converted
    finally:
        converted.close()


class QwenVL(BaseOCR):
    """Wrapper around ``Qwen/Qwen2.5-VL`` for Markdown transcription."""

    name = "qwen_vl"
    supports_batch = True
    supports_async = False

    try:  # pragma: no cover - optional dependency in CI
        import transformers as _transformers

        version = getattr(_transformers, "__version__", "unknown")
    except ModuleNotFoundError:  # pragma: no cover - exercised in tests
        version = "unknown"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        max_new_tokens: int = 256,
        temperature: float | None = 0.1,
        num_beams: int = 1,
        device_map: dict[str, str] | str | None = None,
        max_image_size: int = 1024,
        language_hint: str | None = None,
        allow_network: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_beams = num_beams
        self.device_map = device_map
        self.max_image_size = max_image_size
        self.language_hint = language_hint
        self.allow_network = bool(allow_network)

        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str = "cpu"

    async def recognize_many(
        self,
        items: list[OCRItem],
        *,
        batch_size: int,
        get_logger: Callable[[str, str | None], logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> list[OCRResult]:
        logger = get_logger(__name__, request_id=request_id)
        results: list[OCRResult] = []
        for item in items:
            result = await asyncio.to_thread(self._run_single, item, logger)
            results.append(result)
        return results

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        AutoProcessor = getattr(transformers, "AutoProcessor")
        ModelClass = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")

        cache_dir = get_cached_model(self.model_name, allow_network=self.allow_network)
        processor = AutoProcessor.from_pretrained(cache_dir, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        model = ModelClass.from_pretrained(cache_dir, **model_kwargs)
        model.eval()

        if self.device_map is None:
            device = "cuda" if getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
            model.to(device)
            self._device = device
        elif isinstance(self.device_map, dict):
            self._device = next(iter(self.device_map.values()))
        else:
            self._device = str(self.device_map)

        self._processor = processor
        self._model = model
        self._torch = torch

    def _language(self, item: OCRItem) -> str | None:
        if self.language_hint:
            return self.language_hint
        hint = item.hints.get("language")
        if hint is None:
            return None
        return str(hint)

    def _run_single(self, item: OCRItem, logger: logging.LoggerAdapter) -> OCRResult:
        self._ensure_model()
        assert self._model is not None and self._processor is not None and self._torch is not None

        torch = self._torch
        processor = self._processor
        model = self._model

        start = time.perf_counter()
        provenance = {
            "model": self.name,
            "version": self.version,
            "source_hash": short_hash(item.pdf_path),
            "page_index": item.page_index,
        }

        try:
            with offline_mode(self.allow_network):
                image_path = item.image_path or item.pdf_path
                with _open_image(image_path) as image:
                    image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
                    messages = self._build_messages(item)
                    prompt = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    model_inputs = processor(
                        text=[prompt],
                        images=[image],
                        return_tensors="pt",
                    )
            inputs = {
                key: value.to(self._device)
                if hasattr(value, "to")
                else value
                for key, value in model_inputs.items()
            }
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "num_beams": self.num_beams,
                "do_sample": self.temperature is not None,
            }
            if self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            markdown = text.strip()
        except Exception as exc:  # pragma: no cover - exercised in tests
            elapsed = time.perf_counter() - start
            observe_latency("qwen.page", elapsed)
            logger.exception("Qwen2.5-VL failed for %s", item.pdf_path)
            warning = f"Qwen2.5-VL pipeline error: {exc}"
            return OCRResult(
                markdown="# OCR Failed\n\n> Qwen2.5-VL pipeline error.",
                warnings=[warning],
                time_ms=elapsed * 1000.0,
                provenance=provenance,
            )

        elapsed = time.perf_counter() - start
        observe_latency("qwen.page", elapsed)
        return OCRResult(
            markdown=markdown,
            time_ms=elapsed * 1000.0,
            provenance=provenance,
        )

    def _build_messages(self, item: OCRItem) -> list[dict[str, Any]]:
        language = self._language(item)
        instruction = USER_PROMPT
        if language:
            instruction = f"{USER_PROMPT} Use language hint: {language}."
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            },
        ]


__all__ = ["QwenVL"]


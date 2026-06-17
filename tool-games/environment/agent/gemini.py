import os
import time
from io import BytesIO
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


class GeminiVLClient:
    """
    Lightweight Gemini wrapper matching the tool-games VLM client interface.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant for tool-games physics reasoning.",
        max_retries: int = 3,
        retry_delay: float = 1.5,
        thinking_level: str = "low",
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini backend requires the `google-genai` package. "
                "Install with: pip install google-genai"
            ) from exc

        self._genai = genai
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.thinking_level = (thinking_level or "low").strip().lower()

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key not provided. Set --api_key or GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=resolved_api_key)

    @staticmethod
    def _to_png_bytes(image_object: Any) -> bytes:
        if isinstance(image_object, str):
            if not os.path.isfile(image_object):
                raise FileNotFoundError(f"Image file not found: {image_object}")
            with open(image_object, "rb") as f:
                return f.read()

        if isinstance(image_object, np.ndarray):
            if image_object.max() <= 1.0:
                image_object = (image_object * 255).clip(0, 255).astype(np.uint8)
            else:
                image_object = image_object.astype(np.uint8)
            pil_img = Image.fromarray(image_object)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer.read()

        raise ValueError("Unsupported image object type. Use a file path or NumPy array.")

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""
        chunks: List[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)
        return "".join(chunks).strip()

    def _generate_one(
        self,
        image_bytes: bytes,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        from google.genai import types

        effective_temperature = float(temperature if do_sample else 0.0)
        config = types.GenerateContentConfig(
            temperature=effective_temperature,
            top_p=float(top_p),
            max_output_tokens=int(max_new_tokens),
            system_instruction=self.system_prompt,
        )
        # Lower thinking budget to reduce verbose internal reasoning when supported.
        thinking_budget_by_level = {
            "none": 0,
            "low": 256,
            "medium": 1024,
            "high": 4096,
        }
        budget = thinking_budget_by_level.get(self.thinking_level, thinking_budget_by_level["low"])
        try:
            config.thinking_config = types.ThinkingConfig(thinking_budget=budget)
        except Exception:
            # Some SDK/model versions may not expose thinking_config.
            pass

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part(text=prompt),
                    ],
                    config=config,
                )
                output_text = self._extract_text(response)
                print("[Gemini output begin]")
                print(output_text)
                print("[Gemini output end]")
                return output_text
            except Exception as err:
                last_err = err
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.retry_delay * float(attempt + 1))
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_err}")

    def inference_image(
        self,
        images: Sequence[Any],
        prompts: Sequence[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[List[str], List[Tuple[int | None, int | None]]]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        if len(images) != len(prompts):
            raise ValueError("`images` and `prompts` must have the same length.")

        outputs: List[str] = []
        dims: List[Tuple[int | None, int | None]] = []

        for img, prompt in zip(images, prompts):
            if isinstance(img, np.ndarray):
                dims.append((img.shape[0], img.shape[1]))
            else:
                dims.append((None, None))
            image_bytes = self._to_png_bytes(img)
            output_text = self._generate_one(
                image_bytes=image_bytes,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            outputs.append(output_text)

        return outputs, dims

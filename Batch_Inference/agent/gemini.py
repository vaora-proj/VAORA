"""
Google Gemini vision-language client.

Install:
    pip install google-generativeai pillow numpy

Authentication:
    Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment before running.
"""

import base64
import os
import time
from io import BytesIO
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image


def _numpy_to_pil(image_object) -> Image.Image:
    """Convert a NumPy array (float [0,1] or uint8 [0,255]) to a PIL image."""
    if image_object.max() <= 1.0:
        image_object = (image_object * 255).clip(0, 255).astype(np.uint8)
    else:
        image_object = image_object.astype(np.uint8)
    return Image.fromarray(image_object)


def _load_image_as_pil(image_object) -> Image.Image:
    """Accept a NumPy array, file path, or already-PIL image."""
    if isinstance(image_object, Image.Image):
        return image_object
    if isinstance(image_object, np.ndarray):
        return _numpy_to_pil(image_object)
    if isinstance(image_object, str):
        if image_object.startswith("data:image;base64,"):
            data = base64.b64decode(image_object.split(",", 1)[1])
            return Image.open(BytesIO(data)).convert("RGB")
        return Image.open(image_object).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_object)}")


class GeminiVLClient:
    """
    Gemini vision-language client backed by the google-generativeai SDK.

    Parameters
    ----------
    model_name : str
        Gemini model identifier, e.g. ``"gemini-1.5-flash"`` or
        ``"gemini-2.0-flash"``.
    api_key : str | None
        API key.  Falls back to the ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``
        environment variables when *None*.
    system_prompt : str
        Instruction prepended as a system turn.
    max_retries : int
        Number of automatic retries on transient API errors.
    retry_delay : float
        Base sleep (seconds) between retries; doubles on each attempt.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str | None = None,
        system_prompt: str = "You are a helpful assistant for PHYRE physics reasoning.",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required: pip install google-generativeai"
            ) from e

        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                "or pass api_key=..."
            )

        genai.configure(api_key=resolved_key)
        self._genai = genai
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )

    def _call_single(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        pil_image = _load_image_as_pil(image)
        generation_config = self._genai.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = self._model.generate_content(
                    [pil_image, prompt],
                    generation_config=generation_config,
                )
                return response.text or ""
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"Gemini API error after {self.max_retries} retries: {exc}")
                    return ""
                print(f"Gemini API error (attempt {attempt + 1}): {exc}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
        return ""

    def inference_image(
        self,
        images: Sequence,
        prompts: Sequence[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[str], List[Any]]:
        """
        Run vision inference on a batch of (image, prompt) pairs.

        Note: Gemini API calls are issued sequentially.  Use a higher-level
        async wrapper or a thread pool for parallel throughput if needed.

        Returns
        -------
        output_text : List[str]
            Model responses, one per sample.
        dims : List[None]
            Always None (image dimensions not tracked for API models).
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        if len(images) != len(prompts):
            raise ValueError("`images` and `prompts` must have the same length.")

        output_text = []
        dims: List[Any] = []
        for img, prompt in zip(images, prompts):
            text = self._call_single(img, prompt, max_new_tokens, temperature, top_p)
            output_text.append(text)
            dims.append(None)
        return output_text, dims

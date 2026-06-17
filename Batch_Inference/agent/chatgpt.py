"""
OpenAI ChatGPT vision-language client.

Install:
    pip install openai pillow numpy

Authentication:
    Set OPENAI_API_KEY in the environment before running.
"""

import base64
import os
import time
from io import BytesIO
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image


def _encode_image_to_data_url(image_object) -> str:
    """Return a base-64 PNG data URL from a NumPy array, PIL image, or file path."""
    if isinstance(image_object, str):
        if image_object.startswith("data:image"):
            return image_object
        with open(image_object, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(image_object)[1].lstrip(".").lower() or "png"
        mime = f"image/{ext}"
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    if isinstance(image_object, np.ndarray):
        if image_object.max() <= 1.0:
            image_object = (image_object * 255).clip(0, 255).astype(np.uint8)
        else:
            image_object = image_object.astype(np.uint8)
        pil = Image.fromarray(image_object)
    elif isinstance(image_object, Image.Image):
        pil = image_object
    else:
        raise ValueError(f"Unsupported image type: {type(image_object)}")

    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode()


class ChatGPTVLClient:
    """
    OpenAI ChatGPT vision-language client backed by the openai SDK.

    Parameters
    ----------
    model_name : str
        OpenAI model identifier, e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``.
    api_key : str | None
        API key.  Falls back to ``OPENAI_API_KEY`` environment variable.
    system_prompt : str
        Content for the ``system`` message turn.
    max_retries : int
        Automatic retries on transient API errors.
    retry_delay : float
        Base sleep (seconds) between retries; doubles on each attempt.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        system_prompt: str = "You are a helpful assistant for PHYRE physics reasoning.",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is required: pip install openai") from e

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=..."
            )

        self._client = OpenAI(api_key=resolved_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call_single(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        data_url = _encode_image_to_data_url(image)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"OpenAI API error after {self.max_retries} retries: {exc}")
                    return ""
                print(f"OpenAI API error (attempt {attempt + 1}): {exc}. Retrying in {delay}s…")
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

        Note: OpenAI API calls are issued sequentially.  Wrap in a thread pool
        for parallel throughput if needed.

        Returns
        -------
        output_text : List[str]
            Model responses, one per sample.
        dims : List[None]
            Always None for API-based models.
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

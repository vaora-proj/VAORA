"""
Anthropic Claude vision-language client.

Install:
    pip install anthropic pillow numpy

Authentication:
    Set ANTHROPIC_API_KEY in the environment before running.
"""

import base64
import os
import time
from io import BytesIO
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image


def _encode_image_to_base64(image_object) -> Tuple[str, str]:
    """
    Return (base64_string, media_type) for Anthropic's image content block.

    Accepts: NumPy array, PIL.Image, file path, or existing data URL.
    """
    if isinstance(image_object, str):
        if image_object.startswith("data:image"):
            header, data = image_object.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            return data, mime
        with open(image_object, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(image_object)[1].lstrip(".").lower() or "png"
        mime = f"image/{ext}"
        return base64.b64encode(raw).decode(), mime

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
    return base64.b64encode(buffer.read()).decode(), "image/png"


class ClaudeVLClient:
    """
    Anthropic Claude vision-language client.

    Parameters
    ----------
    model_name : str
        Claude model identifier, e.g. ``"claude-3-5-sonnet-20241022"`` or
        ``"claude-3-opus-20240229"``.
    api_key : str | None
        API key.  Falls back to ``ANTHROPIC_API_KEY`` environment variable.
    system_prompt : str
        System-level instruction.
    max_retries : int
        Automatic retries on transient API errors.
    retry_delay : float
        Base sleep (seconds) between retries; doubles on each attempt.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        system_prompt: str = "You are a helpful assistant for PHYRE physics reasoning.",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic is required: pip install anthropic") from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key=..."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)
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
        b64_data, media_type = _encode_image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model_name,
                    system=self.system_prompt,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # Extract text from the first text content block
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text or ""
                return ""
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"Anthropic API error after {self.max_retries} retries: {exc}")
                    return ""
                print(f"Anthropic API error (attempt {attempt + 1}): {exc}. Retrying in {delay}s…")
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

        Note: Claude API calls are issued sequentially.  Use a thread pool for
        parallel throughput if needed.

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

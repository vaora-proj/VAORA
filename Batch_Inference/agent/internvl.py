"""
InternVL-series vision-language client (local HuggingFace model inference).

Supports InternVL2, InternVL2.5, InternVL3 and compatible variants.

Install:
    pip install transformers torch torchvision pillow numpy

Models (examples):
    OpenGVLab/InternVL2-2B
    OpenGVLab/InternVL2-8B
    OpenGVLab/InternVL2_5-8B
    OpenGVLab/InternVL3-8B
"""

import base64
import os
from io import BytesIO
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# InternVL preprocessing helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    """Return a torchvision transform pipeline for InternVL pixel values."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    Split a large image into tiles (dynamic high-res preprocessing).
    Returns a list of PIL images of size *image_size x image_size*.
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    best_ratio_n, best_ratio_m = 1, 1
    best_ratio_diff = float("inf")
    area = orig_w * orig_h
    for n in range(1, max_num + 1):
        for m in range(1, max_num + 1):
            if n * m > max_num or n * m < min_num:
                continue
            ratio = n / m
            diff = abs(aspect_ratio - ratio)
            if diff < best_ratio_diff or (diff == best_ratio_diff and n * m > best_ratio_n * best_ratio_m):
                best_ratio_diff = diff
                best_ratio_n, best_ratio_m = n, m

    target_w = image_size * best_ratio_n
    target_h = image_size * best_ratio_m
    resized = image.resize((target_w, target_h))

    tiles = []
    for row in range(best_ratio_m):
        for col in range(best_ratio_n):
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
            tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _load_as_pil(image_object) -> Image.Image:
    if isinstance(image_object, Image.Image):
        return image_object.convert("RGB")
    if isinstance(image_object, np.ndarray):
        if image_object.max() <= 1.0:
            image_object = (image_object * 255).clip(0, 255).astype(np.uint8)
        else:
            image_object = image_object.astype(np.uint8)
        return Image.fromarray(image_object).convert("RGB")
    if isinstance(image_object, str):
        if image_object.startswith("data:image;base64,"):
            data = base64.b64decode(image_object.split(",", 1)[1])
            return Image.open(BytesIO(data)).convert("RGB")
        return Image.open(image_object).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_object)}")


def _preprocess_image(
    image_object, max_num: int = 12, input_size: int = 448
) -> Tuple[torch.Tensor, int]:
    """
    Return (pixel_values, num_patches) for a single image.

    pixel_values shape: (num_patches, 3, input_size, input_size)
    """
    transform = _build_transform(input_size)
    pil = _load_as_pil(image_object)
    tiles = _dynamic_preprocess(pil, max_num=max_num, image_size=input_size, use_thumbnail=True)
    pixel_values = torch.stack([transform(t) for t in tiles])  # (N, 3, H, W)
    return pixel_values, len(tiles)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class InternVLClient:
    """
    InternVL-series local model client.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"OpenGVLab/InternVL2-8B"``.
    local_load_path : str | None
        Path to a local checkpoint directory (overrides *model_name* if set).
    device_map : str | None
        ``transformers`` device-map string.  Defaults to ``"auto"``.
    torch_dtype : str | torch.dtype
        Compute dtype.  ``"auto"`` lets the framework decide.
    use_flash_attention : bool
        Enable ``flash_attention_2`` (requires compatible GPU + package).
    max_num_tiles : int
        Maximum image tiles for dynamic preprocessing (higher = more detail).
    input_size : int
        Tile size in pixels (default 448 for InternVL2).
    system_prompt : str
        System instruction prepended to every conversation.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-8B",
        local_load_path: str | None = None,
        device_map: str | None = "auto",
        torch_dtype: str | torch.dtype = "auto",
        use_flash_attention: bool = False,
        max_num_tiles: int = 12,
        input_size: int = 448,
        system_prompt: str = "You are a helpful assistant for PHYRE physics reasoning.",
    ):
        from transformers import AutoModel, AutoTokenizer

        model_path = local_load_path or model_name

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if use_flash_attention:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            elif torch_dtype != "auto":
                model_kwargs["torch_dtype"] = torch_dtype
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.system_prompt = system_prompt
        self.max_num_tiles = max_num_tiles
        self.input_size = input_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def inference_image(
        self,
        images: Sequence,
        prompts: Sequence[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[str], List[Tuple[int | None, int | None]]]:
        """
        Run batch inference using InternVL's ``batch_chat`` API.

        Returns
        -------
        output_text : List[str]
        dims : List[Tuple[int, int] | None]
            (height, width) for NumPy inputs; None otherwise.
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        if len(images) != len(prompts):
            raise ValueError("`images` and `prompts` must have the same length.")

        dtype = next(self.model.parameters()).dtype
        all_pixel_values: List[torch.Tensor] = []
        num_patches_list: List[int] = []
        dims: List[Any] = []

        for img in images:
            if isinstance(img, np.ndarray):
                dims.append((img.shape[0], img.shape[1]))
            else:
                dims.append(None)
            pv, n_patches = _preprocess_image(img, max_num=self.max_num_tiles, input_size=self.input_size)
            all_pixel_values.append(pv.to(dtype).to(self.device))
            num_patches_list.append(n_patches)

        # Prepend <image> placeholder to each prompt so InternVL maps the tiles.
        questions = [f"<image>\n{p}" for p in prompts]

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Stack into a single tensor along the batch dimension for batch_chat.
        pixel_values_batch = torch.cat(all_pixel_values, dim=0)

        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values_batch,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
            history=None,
            return_history=False,
        )

        if isinstance(responses, (list, tuple)):
            output_text = [str(r) for r in responses]
        else:
            output_text = [str(responses)]

        return output_text, dims

import os
import base64
from io import BytesIO
from typing import List, Sequence, Tuple
from typing import Optional
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# Provided by the Qwen stack; handles multi-modal input preprocessing.
from qwen_vl_utils import process_vision_info


class Qwen3VLClient:
    """
    Lightweight Qwen3-VL wrapper for tool-games usage.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-3B-Instruct",
        local_load_path: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[str] = "auto",
        use_flash_attention: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        system_prompt: str = "You are a helpful assistant for tool-games physics reasoning.",
    ):
        model_kwargs = {"device_map": device_map, "trust_remote_code": True}
        if use_flash_attention:
            model_kwargs.update(
                {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
            )
        else:
            model_kwargs.update({"torch_dtype": torch_dtype, "attn_implementation": "sdpa"})

        model_path = local_load_path or model_name
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=min_pixels if min_pixels is not None else 4 * 28 * 28,
            max_pixels=max_pixels if max_pixels is not None else 16384 * 28 * 28,
        )
        self.system_prompt = system_prompt

    @staticmethod
    def encode_image(image_object) -> str:
        if isinstance(image_object, str) and os.path.isfile(image_object):
            with open(image_object, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image_object, np.ndarray):
            if image_object.max() <= 1.0:
                image_object = (image_object * 255).clip(0, 255).astype(np.uint8)
            else:
                image_object = image_object.astype(np.uint8)
            img_pil = Image.fromarray(image_object)
            buffer = BytesIO()
            img_pil.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        else:
            raise ValueError("Unsupported image object type. Use a file path or NumPy array.")
        return f"data:image;base64,{image_base64}"

    @staticmethod
    def _place_input_image(
        text: str,
        image_pad: str = "<|vision_start|><|image_pad|><|vision_end|>",
        image_placeholder: str = "<image>",
    ) -> str:
        # Keep chat-template image tokens unless caller explicitly uses <image>.
        if image_placeholder in text:
            text = text.replace(image_pad, "")
            text = text.replace(image_placeholder, image_pad)
        return text

    @staticmethod
    def _to_device(batch_encoding: dict, device: torch.device | str):
        moved = {}
        for key, val in batch_encoding.items():
            if isinstance(val, torch.Tensor):
                moved[key] = val.to(device)
            elif isinstance(val, (list, tuple)):
                new_list = []
                for item in val:
                    new_list.append(item.to(device) if isinstance(item, torch.Tensor) else item)
                moved[key] = new_list
            else:
                moved[key] = val
        return moved

    def inference_image(
        self,
        images: Sequence,
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

        processed_images = []
        dims = []
        for img in images:
            if isinstance(img, np.ndarray):
                dims.append((img.shape[0], img.shape[1]))
                processed_images.append(self.encode_image(img))
            elif isinstance(img, str) and not img.startswith("data:image;base64,"):
                dims.append(None)
                processed_images.append(self.encode_image(img))
            else:
                dims.append(None)
                processed_images.append(img)

        messages = []
        for img, prompt in zip(processed_images, prompts):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            )

        text_inputs = [
            self.processor.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    msg,
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in messages
        ]
        text_inputs = [self._place_input_image(txt) for txt in text_inputs]
        image_inputs, video_inputs = process_vision_info(messages)

        batch = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch = self._to_device(batch, self.model.device)

        sample_enabled = bool(do_sample and temperature is not None and temperature > 0)
        generate_kwargs = {
            **batch,
            "max_new_tokens": max_new_tokens,
            "do_sample": sample_enabled,
            "pad_token_id": getattr(self.processor.tokenizer, "eos_token_id", None),
            "eos_token_id": getattr(self.processor.tokenizer, "eos_token_id", None),
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }
        if sample_enabled:
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)

        generated_ids = self.model.generate(**generate_kwargs)

        input_ids_tensor = batch.get("input_ids")
        if isinstance(input_ids_tensor, torch.Tensor):
            trimmed = [
                out_ids[input_ids_tensor.shape[1] :]
                if out_ids.shape[0] > input_ids_tensor.shape[1]
                else out_ids
                for out_ids in generated_ids
            ]
        else:
            trimmed = generated_ids

        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        return output_text, dims

# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import copy
import logging
import os
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop import AgentLoopBase
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("partial_single_turn_agent")
class PartialSingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        output: Optional[AgentLoopOutput] = kwargs.get("output", None)
        messages = list(kwargs["raw_prompt"])
        messages_with_images, image_data_from_prompt = self._extract_and_encode_images(messages)
        param_version = kwargs.get("param_version", 0)

        metrics = {}
        request_id = uuid4().hex
        # Prefer image data encoded from the prompt; fall back to any provided multi_modal_data
        image_data = image_data_from_prompt or (copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None)))
        multi_modal_data = {"images": image_data} if image_data is not None else {}

        param_version_start = param_version
        param_version_end = param_version

        if not output:
            # Use processor to preserve multimodal prompts when available; fall back to tokenizer otherwise.
            use_correct_processor = self.processor is not None
            if self.processor is not None and use_correct_processor:

                def get_prompt_ids():
                    raw_prompt = self.processor.apply_chat_template(
                        messages_with_images,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    )
                    model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
                    return model_inputs.pop("input_ids").squeeze(0).tolist()

                prompt_ids = await self.loop.run_in_executor(None, get_prompt_ids)
            else:
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages_with_images,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
        else:
            if output.extra_fields.get("is_cancel", False):
                # Resume the paused sample,
                # add the result directly after prompt_ids,
                # and reset generate_sequences metric
                prompt_ids = output.prompt_ids + output.response_ids
                metrics["generate_sequences"] = output.metrics.generate_sequences
                param_version_start = output.extra_fields.get("param_version_start", param_version)
            else:
                # In the same batch of samples,
                # some are canceled and some are not.
                # The samples without partial rollout are returned directly.
                return output
        with simple_timer("generate_sequences", metrics):
            response_ids, response_logprobs, is_cancel = await self.server_manager.generate_for_partial(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
            )
        if not output:
            response_mask = [1] * len(response_ids)
        else:
            # Pause the sample to be resumed, add the output result to response_ids, and reset response_mask
            prompt_ids = output.prompt_ids
            response_logprobs = output.response_logprobs + response_logprobs
            response_ids = output.response_ids + response_ids
            response_mask = [1] * len(response_ids)
        if len(response_ids) >= self.response_length:
            is_cancel = False

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length],
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "is_cancel": is_cancel,
                "param_version_start": param_version_start,
                "param_version_end": param_version_end,
            },
            multi_modal_data=multi_modal_data,
        )

    @staticmethod
    def _extract_and_encode_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
        """Extract images from messages, encode them to base64 data URLs, and return updated messages."""

        def encode_image(content_image: Any) -> Optional[str]:
            try:
                if isinstance(content_image, (bytes, bytearray)):
                    raw_bytes = bytes(content_image)
                else:
                    buffer = BytesIO()
                    content_image.save(buffer, format="PNG")
                    raw_bytes = buffer.getvalue()
                return f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"
            except Exception as err:  # pragma: no cover - defensive
                logger.warning("Failed to encode image content to base64: %s", err)
                return None

        processed_messages: list[dict[str, Any]] = copy.deepcopy(messages)
        encoded_images: list[Any] = []

        for msg in processed_messages:
            contents = msg.get("content", [])
            new_contents = []
            for content in contents:
                if isinstance(content, dict) and content.get("type") == "image" and "image" in content:
                    image_url = encode_image(content["image"])
                    if image_url is not None:
                        encoded_images.append(image_url)
                        content = {k: v for k, v in content.items() if k != "image"}
                        content["image_url"] = image_url
                new_contents.append(content)
            msg["content"] = new_contents

        return processed_messages, encoded_images

"""OpenAI-compatible multimodal LLM backbone used by all reasoning modules."""

from __future__ import annotations

import base64
import os
import random
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_BASE_URL = "https://ai.juguang.chat/v1"
DEFAULT_MODEL_NAME = "gemini-3-flash-preview-thinking"


class MLLMBackbone:
    """Thin wrapper around an OpenAI-compatible chat-completions API.

    API credentials are read from environment variables by default:
    - MLLM_API_KEY
    - MLLM_BASE_URL
    - MLLM_MODEL_NAME
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        min_interval: float = 0.5,
        max_tokens: int = 4096,
    ) -> None:
        from openai import APIConnectionError, APIError, OpenAI, RateLimitError

        self.api_key = api_key if api_key is not None else os.getenv("MLLM_API_KEY", "")
        self.base_url = base_url or os.getenv("MLLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("MLLM_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.max_tokens = max_tokens
        self._last_request_time = 0.0
        self._min_interval = min_interval
        self._rate_limit_error = RateLimitError
        self._api_error = APIError
        self._api_connection_error = APIConnectionError
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _wait_for_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def pil_to_base64(img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _create_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        max_retries: int,
        base_delay: float,
        operation_name: str,
    ) -> str:
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError(f"{operation_name} returned an empty response.")
                return content
            except self._rate_limit_error:
                wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"[RateLimitError] {operation_name} attempt {attempt + 1}, wait {wait_time:.1f}s...")
                time.sleep(wait_time)
            except (self._api_error, self._api_connection_error) as exc:
                wait_time = base_delay + random.uniform(0, 1)
                print(f"[API/Network Error] {operation_name} attempt {attempt + 1}, wait {wait_time:.1f}s...")
                print(exc)
                time.sleep(wait_time)
        raise RuntimeError(f"{operation_name} failed after {max_retries} retries.")

    def generate_response_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        return self._create_with_retry(
            messages,
            max_retries=5,
            base_delay=5,
            operation_name="generate_response_text",
        )

    def generate_response_multimodal_single(self, image: Image.Image, prompt: str) -> str:
        if image is None:
            raise ValueError("image must be a PIL image.")
        encoded_image = self.pil_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                ],
            }
        ]
        return self._create_with_retry(
            messages,
            max_retries=10,
            base_delay=7,
            operation_name="generate_response_multimodal_single",
        )

    def generate_response_multimodal_multi(self, images: list[Image.Image], prompt: str) -> str:
        if not images or len(images) < 2:
            raise ValueError("images list must contain at least two PIL images.")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            if image is None:
                raise ValueError("images must be PIL images; got None.")
            encoded_image = self.pil_to_base64(image)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})
        messages = [{"role": "user", "content": content}]
        return self._create_with_retry(
            messages,
            max_retries=10,
            base_delay=7,
            operation_name="generate_response_multimodal_multi",
        )

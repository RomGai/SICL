"""Image generation module placeholder."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.schemas import GenerationPromptSpec


class ImageGenerationModule:
    """Stub for future image generation backends.

    This is intentionally the only module that does not use MLLMBackbone.
    Replace or subclass this class to connect a real reference-image-conditioned generator.
    """

    def generate(
        self,
        original_image: Image.Image,
        generation_prompt_spec: GenerationPromptSpec,
    ) -> Image.Image | None:
        _ = (original_image, generation_prompt_spec)
        raise NotImplementedError("Image generation backend is not implemented yet.")

"""Image generation modules.

The default module remains a no-op stub, while ``QwenImageEditGenerationModule``
provides an optional local Diffusers backend for reference-image-conditioned image
editing/generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.schemas import GenerationPromptSpec

ImageGenerationBackend = Literal["stub", "qwen_edit"]


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


@dataclass
class QwenImageEditConfig:
    """Runtime configuration for the optional Qwen image edit backend."""

    model_name: str = "Qwen/Qwen-Image-Edit-2511"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    seed: int = 0
    true_cfg_scale: float = 4.0
    negative_prompt: str = " "
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    num_images_per_prompt: int = 1


class QwenImageEditGenerationModule(ImageGenerationModule):
    """Optional Diffusers/Qwen reference-image-conditioned generation backend.

    This module follows the user-provided Qwen-Image-Edit flow but keeps all heavy
    dependencies lazy. It is only imported/initialized when requested, so the base
    dry-run pipeline remains lightweight.
    """

    def __init__(self, config: QwenImageEditConfig | None = None) -> None:
        self.config = config or QwenImageEditConfig()
        self._pipeline = None
        self._torch = None

    def _resolve_torch_dtype(self, torch_module: object) -> object:
        dtype = self.config.torch_dtype
        if dtype == "bfloat16":
            return torch_module.bfloat16
        if dtype == "float16":
            return torch_module.float16
        if dtype == "float32":
            return torch_module.float32
        raise ValueError(
            "Unsupported qwen torch dtype "
            f"{dtype!r}; expected one of: bfloat16, float16, float32."
        )

    def _load_pipeline(self) -> object:
        if self._pipeline is not None:
            return self._pipeline

        import importlib.util

        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("diffusers") is None:
            raise ImportError(
                "Qwen image generation requires optional dependencies. Install them with:\n"
                "  pip install git+https://github.com/huggingface/diffusers\n"
                "  pip install torch Pillow"
            )

        import torch
        from diffusers import QwenImageEditPlusPipeline

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=self._resolve_torch_dtype(torch),
        )
        pipeline.to(self.config.device)
        pipeline.set_progress_bar_config(disable=None)
        self._pipeline = pipeline
        self._torch = torch
        return pipeline

    def generate(
        self,
        original_image: Image.Image,
        generation_prompt_spec: GenerationPromptSpec,
    ) -> Image.Image:
        if original_image is None:
            raise ValueError("original_image must be a PIL image for qwen_edit generation.")

        pipeline = self._load_pipeline()
        torch = self._torch
        if torch is None:
            raise RuntimeError("Qwen image edit pipeline loaded without torch module state.")

        prompt = generation_prompt_spec.image_generation_prompt
        inputs = {
            "image": [original_image],
            "prompt": prompt,
            "generator": torch.manual_seed(self.config.seed),
            "true_cfg_scale": self.config.true_cfg_scale,
            "negative_prompt": self.config.negative_prompt,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "num_images_per_prompt": self.config.num_images_per_prompt,
        }
        with torch.inference_mode():
            output = pipeline(**inputs)
        if not getattr(output, "images", None):
            raise RuntimeError("Qwen image edit pipeline returned no images.")
        return output.images[0]


def create_image_generation_module(
    backend: ImageGenerationBackend = "stub",
    *,
    qwen_config: QwenImageEditConfig | None = None,
) -> ImageGenerationModule:
    """Factory for optional image-generation backends.

    Parameters
    ----------
    backend:
        ``"stub"`` keeps the placeholder behavior. ``"qwen_edit"`` enables the
        optional Diffusers Qwen-Image-Edit backend.
    qwen_config:
        Optional Qwen runtime configuration used only when ``backend="qwen_edit"``.
    """
    if backend == "stub":
        return ImageGenerationModule()
    if backend == "qwen_edit":
        return QwenImageEditGenerationModule(config=qwen_config)
    raise ValueError(f"Unknown image generation backend: {backend!r}")

"""CLI entry point for running the synthetic ICL pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.modules.image_generation import QwenImageEditConfig, create_image_generation_module
from synthetic_icl.pipeline import SyntheticICLPipeline
from synthetic_icl.schemas import SyntheticExample


def _print_json(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _save_generated_images(examples: list[SyntheticExample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, example in enumerate(examples):
        if example.image is None:
            continue
        safe_scenario_id = example.scenario.scenario_id or f"example_{idx:03d}"
        image_path = output_dir / f"{idx:03d}_{safe_scenario_id}.png"
        example.image.save(image_path)
        example.verification_result.setdefault("saved_image_path", str(image_path.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic multimodal ICL pipeline.")
    parser.add_argument("--image", required=True, help="Path to the original image.")
    parser.add_argument("--query", required=True, help="Original query. It will not be rewritten.")
    parser.add_argument("--num-scenarios", type=int, default=5, help="Number of scenarios to expand.")
    parser.add_argument("--num-answers-per-scenario", type=int, default=1, help="Answers per scenario.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of selected examples.")
    parser.add_argument(
        "--image-generation-pipe",
        choices=["stub", "qwen_edit"],
        default="stub",
        help="Image generation backend. Use 'qwen_edit' to run Qwen-Image-Edit generation.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Skip real image generation. Defaults to true for --image-generation-pipe stub "
            "and false for --image-generation-pipe qwen_edit."
        ),
    )
    parser.add_argument("--output-dir", default="synthetic_outputs", help="Directory for generated images.")
    parser.add_argument("--qwen-device", default="cuda", help="Device for qwen_edit, e.g. cuda or cpu.")
    parser.add_argument(
        "--qwen-torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for qwen_edit.",
    )
    parser.add_argument("--qwen-seed", type=int, default=0, help="Random seed for qwen_edit generation.")
    parser.add_argument("--qwen-steps", type=int, default=40, help="Inference steps for qwen_edit generation.")
    parser.add_argument("--qwen-true-cfg-scale", type=float, default=4.0, help="true_cfg_scale for qwen_edit.")
    parser.add_argument("--qwen-guidance-scale", type=float, default=1.0, help="guidance_scale for qwen_edit.")
    parser.add_argument("--qwen-negative-prompt", default=" ", help="Negative prompt for qwen_edit.")
    parser.add_argument("--verbose", action="store_true", help="Print pipeline stage progress and intermediate results.")
    args = parser.parse_args()

    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise ImportError("Pillow is required to load images. Install it with: pip install Pillow")

    from PIL import Image

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        original_image = img.convert("RGB")

    dry_run = args.dry_run if args.dry_run is not None else args.image_generation_pipe != "qwen_edit"
    qwen_config = QwenImageEditConfig(
        device=args.qwen_device,
        torch_dtype=args.qwen_torch_dtype,
        seed=args.qwen_seed,
        true_cfg_scale=args.qwen_true_cfg_scale,
        negative_prompt=args.qwen_negative_prompt,
        num_inference_steps=args.qwen_steps,
        guidance_scale=args.qwen_guidance_scale,
    )
    image_generation_module = create_image_generation_module(
        args.image_generation_pipe,
        qwen_config=qwen_config,
    )

    backbone = MLLMBackbone()
    pipeline = SyntheticICLPipeline(backbone, image_generation_module=image_generation_module)
    selected_examples = pipeline.run(
        original_image=original_image,
        original_query=args.query,
        num_scenarios=args.num_scenarios,
        num_answers_per_scenario=args.num_answers_per_scenario,
        top_k=args.top_k,
        dry_run=dry_run,
        verbose=args.verbose,
    )

    if not dry_run:
        _save_generated_images(pipeline.last_candidates, Path(args.output_dir))

    if selected_examples:
        _print_json("TaskIR", selected_examples[0].task_ir.to_dict())

    _print_json("ScenarioSpecs", [example.scenario.to_dict() for example in pipeline.last_candidates])
    _print_json("AnswerSpecs", [example.answer_spec.to_dict() for example in pipeline.last_candidates])
    _print_json(
        "Generation Prompts",
        [asdict(example.generation_prompt) for example in pipeline.last_candidates],
    )
    _print_json("Selected Examples Metadata", [example.to_metadata_dict() for example in selected_examples])


if __name__ == "__main__":
    main()

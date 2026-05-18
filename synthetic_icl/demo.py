"""Minimal dry-run demo for the synthetic ICL pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.pipeline import SyntheticICLPipeline


def _print_json(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dry-run synthetic multimodal ICL pipeline demo.")
    parser.add_argument("--image", required=True, help="Path to the original image.")
    parser.add_argument("--query", required=True, help="Original query. It will not be rewritten.")
    parser.add_argument("--num-scenarios", type=int, default=5, help="Number of scenarios to expand.")
    parser.add_argument("--num-answers-per-scenario", type=int, default=1, help="Answers per scenario.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of selected examples.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        original_image = img.convert("RGB")

    backbone = MLLMBackbone()
    pipeline = SyntheticICLPipeline(backbone)
    selected_examples = pipeline.run(
        original_image=original_image,
        original_query=args.query,
        num_scenarios=args.num_scenarios,
        num_answers_per_scenario=args.num_answers_per_scenario,
        top_k=args.top_k,
        dry_run=True,
    )

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

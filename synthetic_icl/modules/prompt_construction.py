"""Generation prompt construction module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import AnswerSpec, GenerationPromptSpec, ScenarioSpec, TaskIR


class GenerationPromptConstructionModule:
    """Build prompts for a future reference-image-conditioned generator."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(
        self,
        original_image: Image.Image,
        task_ir: TaskIR,
        scenario: ScenarioSpec,
        answer_spec: AnswerSpec,
        original_query: str,
    ) -> GenerationPromptSpec:
        _ = original_image
        prompt = f"""
You are writing an image-generation prompt for a reference-image-conditioned image generator.

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

ScenarioSpec:
{json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2)}

AnswerSpec:
{json.dumps(answer_spec.to_dict(), ensure_ascii=False, indent=2)}

Original query that must remain unchanged:
{json.dumps(original_query, ensure_ascii=False)}

Construct a GenerationPromptSpec.

The image_generation_prompt MUST explicitly require all of the following:
1. Use the original image as a visual reference for task-related style, layout, visual organization, labels, relationships, or expression mode.
2. Do not copy the original image's exact content, objects, text, data values, or scene specifics.
3. Generate a new image in the new scenario.
4. The new image must be answerable by the exact original_query, with no new or rewritten query.
5. The correct known answer must be exactly AnswerSpec.answer.
6. The visual evidence must clearly support the known answer and avoid ambiguity.

Return ONLY strict JSON with schema:
{{
  "scenario_id": string,
  "original_query": string,
  "known_answer": string,
  "image_generation_prompt": string,
  "reference_policy": string,
  "must_include": [string],
  "must_avoid": [string]
}}
""".strip()
        raw = self.backbone.generate_response_text(prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("GenerationPromptConstructionModule expected a JSON object.")
        parsed["scenario_id"] = scenario.scenario_id
        parsed["original_query"] = original_query
        parsed["known_answer"] = answer_spec.answer
        parsed.setdefault(
            "reference_policy",
            "Use original_image only as a task-related visual reference; do not copy concrete content.",
        )
        return GenerationPromptSpec.from_dict(parsed)

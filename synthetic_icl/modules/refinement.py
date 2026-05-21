"""Prompt refinement module for iterative image editing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import AnswerSpec, GenerationPromptSpec, ScenarioSpec, TaskIR


class ImageRefinementPromptModule:
    """Generate edit prompts to align generated images with task/style constraints."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(
        self,
        original_image: Image.Image,
        synthetic_image: Image.Image,
        evaluation_query: str,
        source_query: str,
        task_ir: TaskIR,
        scenario: ScenarioSpec,
        answer_spec: AnswerSpec,
        base_prompt_spec: GenerationPromptSpec,
        verification_result: dict,
        attempt_history: list[dict] | None = None,
        history_images: list[Image.Image] | None = None,
    ) -> str:
        prompt = f"""
You are improving a synthetic multimodal ICL sample via image editing.

Evaluation query for current synthetic sample (must remain unchanged during this edit):
{json.dumps(evaluation_query, ensure_ascii=False)}

Source/original query for task-alignment reference:
{json.dumps(source_query, ensure_ascii=False)}

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

ScenarioSpec:
{json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2)}

AnswerSpec:
{json.dumps(answer_spec.to_dict(), ensure_ascii=False, indent=2)}

Current generation prompt:
{json.dumps(base_prompt_spec.image_generation_prompt, ensure_ascii=False)}

Verification result of current synthetic image:
{json.dumps(verification_result, ensure_ascii=False, indent=2)}

Compact edit history summary (latest few; do not repeat ineffective edits):
{json.dumps((attempt_history or [])[-3:], ensure_ascii=False, indent=2)}

You are given an ordered image sequence: image A is original reference, middle images are prior attempts, and final image is current synthetic image to edit.
Infer trajectory of what changed across attempts and avoid repeating ineffective edits.

Output an image-edit prompt that:
- keeps B answerable by the exact query,
- preserves known answer={json.dumps(answer_spec.answer, ensure_ascii=False)},
- fixes verification issues,
- moves style/distribution closer to A without copying concrete scene content.

Return ONLY strict JSON:
{{
  "edit_prompt": string,
  "focus_changes": [string]
}}
""".strip()
        images = [original_image]
        for img in history_images or []:
            if img is not None:
                images.append(img)
        images.append(synthetic_image)
        raw = self.backbone.generate_response_multimodal_multi(images, prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ImageRefinementPromptModule expected JSON object.")
        edit_prompt = str(parsed.get("edit_prompt", "")).strip()
        if not edit_prompt:
            raise ValueError("ImageRefinementPromptModule returned empty edit_prompt.")
        return edit_prompt

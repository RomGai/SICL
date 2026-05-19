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
        original_query: str,
        task_ir: TaskIR,
        scenario: ScenarioSpec,
        answer_spec: AnswerSpec,
        base_prompt_spec: GenerationPromptSpec,
        verification_result: dict,
    ) -> str:
        prompt = f"""
You are improving a synthetic multimodal ICL sample via image editing.

Exact query (must remain unchanged):
{json.dumps(original_query, ensure_ascii=False)}

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

Given image A=original reference image and image B=current synthetic image, output an image-edit prompt that:
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
        raw = self.backbone.generate_response_multimodal_multi([original_image, synthetic_image], prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ImageRefinementPromptModule expected JSON object.")
        edit_prompt = str(parsed.get("edit_prompt", "")).strip()
        if not edit_prompt:
            raise ValueError("ImageRefinementPromptModule returned empty edit_prompt.")
        return edit_prompt

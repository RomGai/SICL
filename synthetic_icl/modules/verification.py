"""Synthetic image verification module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import AnswerSpec, ScenarioSpec, TaskIR


class VerificationModule:
    """Verify generated images against the unchanged query and known answer."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(
        self,
        synthetic_image: Image.Image | None,
        original_query: str,
        known_answer: str,
        task_ir: TaskIR,
        scenario: ScenarioSpec,
        answer_spec: AnswerSpec,
    ) -> dict[str, Any]:
        if synthetic_image is None:
            return {
                "status": "skipped",
                "pass": None,
                "predicted_answer": None,
                "matches_known_answer": None,
                "ambiguity_score": None,
                "issues": ["synthetic_image is None; dry_run or image generation stub was used."],
                "reason": "Verification skipped because no synthetic image is available.",
            }

        prompt = f"""
You are verifying one synthetic multimodal ICL demonstration image.

Exact query to answer from the attached synthetic image:
{json.dumps(original_query, ensure_ascii=False)}

Known planned answer:
{json.dumps(known_answer, ensure_ascii=False)}

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

ScenarioSpec:
{json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2)}

AnswerSpec:
{json.dumps(answer_spec.to_dict(), ensure_ascii=False, indent=2)}

Check:
- Can the exact original_query be answered from this image without rewriting the query?
- What answer does the image support?
- Does it match the known planned answer?
- Is there ambiguity or missing visual evidence?
- Are required labels/entities/attributes/relations visible?
- Does the image appear to copy the original image's concrete content instead of only task-related style/layout?

Return ONLY strict JSON:
{{
  "status": "completed",
  "pass": true,
  "predicted_answer": string,
  "matches_known_answer": true,
  "ambiguity_score": 0.0,
  "issues": [string],
  "reason": string
}}
""".strip()
        raw = self.backbone.generate_response_multimodal_single(synthetic_image, prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("VerificationModule expected a JSON object.")
        parsed.setdefault("status", "completed")
        return parsed

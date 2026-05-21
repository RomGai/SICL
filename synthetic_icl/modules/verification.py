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
    """Verify generated images and provide control-flow decisions."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(
        self,
        synthetic_image: Image.Image | None,
        evaluation_query: str,
        known_answer: str,
        task_ir: TaskIR,
        scenario: ScenarioSpec,
        answer_spec: AnswerSpec,
        source_query: str | None = None,
        original_image: Image.Image | None = None,
        verify_against_original: bool = False,
        attempt_history: list[dict[str, Any]] | None = None,
        history_images: list[Image.Image] | None = None,
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
                "is_valid_demo": False,
                "is_good_enough": False,
                "failure_type": "image_unavailable",
                "recommended_action": "regenerate",
                "confidence": 0.0,
                "edit_targets": [],
            }

        original_alignment_policy = (
            """
Additional requirement:
- Compare the candidate against the ORIGINAL reference image and original task context.
- Check whether visual distribution, task expression style, and task type remain broadly aligned.
- If distribution/task-type drift is severe, avoid "accept"; prefer "edit" or "regenerate".
"""
            if verify_against_original and original_image is not None
            else ""
        )
        sequence_instructions = (
            """
Image sequence protocol:
- The FIRST attached image is the ORIGINAL reference image for task/distribution alignment.
- Any middle images are prior synthetic attempts (history only).
- The LAST attached image is the CURRENT candidate to evaluate.
- Do NOT score the first/middle images as candidates; use them only as reference context.
"""
            if verify_against_original and original_image is not None
            else """
Image sequence protocol:
- If multiple images are attached, treat earlier images as prior synthetic attempts (history only).
- The LAST attached image is the CURRENT candidate to evaluate.
- Do NOT score earlier images as candidates.
"""
        )
        prompt = f"""
You are verifying one synthetic multimodal ICL demonstration image.

Evaluation query to answer from the attached synthetic image:
{json.dumps(evaluation_query, ensure_ascii=False)}

Source/original query for task-alignment reference:
{json.dumps(source_query or task_ir.original_query, ensure_ascii=False)}

Known planned answer:
{json.dumps(known_answer, ensure_ascii=False)}

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

ScenarioSpec:
{json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2)}

AnswerSpec:
{json.dumps(answer_spec.to_dict(), ensure_ascii=False, indent=2)}

Compact attempt history (latest few; use as context, avoid repeating failed edits):
{json.dumps((attempt_history or [])[-3:], ensure_ascii=False, indent=2)}

{sequence_instructions}

Return ONLY strict JSON:
{{
  "status": "completed",
  "pass": true,
  "predicted_answer": string,
  "matches_known_answer": true,
  "ambiguity_score": 0.0,
  "issues": [string],
  "reason": string,
  "is_valid_demo": true,
  "is_good_enough": false,
  "failure_type": "none",
  "recommended_action": "accept",
  "confidence": 0.9,
  "edit_targets": [string]
}}

Decision policy:
- recommended_action="regenerate" when answer mismatch / missing critical evidence / severe ambiguity makes this candidate unusable.
- recommended_action="edit" when candidate is valid but needs style/layout/entity detail correction.
- recommended_action="accept" when it is already good enough as an ICL demonstration.
- is_valid_demo should indicate whether this image can be kept as a usable demonstration at all.
- is_good_enough should be true only when no further editing is needed.
{original_alignment_policy}
""".strip()
        images = []
        if verify_against_original and original_image is not None:
            images.append(original_image)
        images.extend([img for img in (history_images or []) if img is not None])
        images.append(synthetic_image)
        if len(images) >= 2:
            raw = self.backbone.generate_response_multimodal_multi(images, prompt)
        else:
            raw = self.backbone.generate_response_multimodal_single(synthetic_image, prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("VerificationModule expected a JSON object.")
        parsed.setdefault("status", "completed")
        parsed.setdefault("is_valid_demo", bool(parsed.get("pass")))
        parsed.setdefault("is_good_enough", bool(parsed.get("pass")))
        parsed.setdefault("failure_type", "none" if bool(parsed.get("pass")) else "other")
        parsed.setdefault("recommended_action", "accept" if bool(parsed.get("pass")) else "regenerate")
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("edit_targets", [])
        return parsed

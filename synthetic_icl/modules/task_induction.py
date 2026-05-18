"""Task induction module."""

from __future__ import annotations

import json
from typing import Any

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import TaskIR


class TaskInductionModule:
    """Convert concrete understanding into a reusable task IR."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(self, original_query: str, understanding_result: dict[str, Any]) -> TaskIR:
        prompt = f"""
You are designing a TaskIR for multimodal in-context learning demonstrations.

Input original_query (must remain byte-for-byte unchanged in all synthetic examples):
{json.dumps(original_query, ensure_ascii=False)}

Image-query understanding JSON:
{json.dumps(understanding_result, ensure_ascii=False, indent=2)}

Induce a general task representation that supports generating NEW images where the SAME original_query can be asked.

Rules:
- Do NOT produce a new question.
- query_invariance_rule must explicitly state that synthetic examples must use exactly original_query.
- candidate_answer_space should contain plausible labels/values for answers whenever possible.
- image_generation_requirements must describe task-relevant style/layout/evidence to preserve while avoiding copying exact original content.
- verification_criteria must be actionable checks for later validation.

Return ONLY strict JSON matching this schema:
{{
  "original_query": string,
  "image_type": string,
  "scene_summary": string,
  "task_family": string,
  "reasoning_type": string,
  "target_entities": [string],
  "reference_entities": [string],
  "queried_attributes": [string],
  "required_visual_evidence": [string],
  "answer_type": "yes_no|number|entity_choice|region_choice|text|attribute_value|free_form",
  "candidate_answer_space": [string],
  "query_invariance_rule": string,
  "image_generation_requirements": [string],
  "ambiguity_risks": [string],
  "verification_criteria": [string]
}}
""".strip()
        raw = self.backbone.generate_response_text(prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("TaskInductionModule expected a JSON object.")
        parsed["original_query"] = original_query
        if "query_invariance_rule" not in parsed or original_query not in parsed.get("query_invariance_rule", ""):
            parsed["query_invariance_rule"] = (
                "Every synthetic example query must exactly equal original_query, with no rewriting: "
                f"{original_query}"
            )
        return TaskIR.from_dict(parsed)

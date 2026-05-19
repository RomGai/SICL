"""Image-query understanding module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse


class ImageQueryUnderstandingModule:
    """Understand the original image and query into a structured draft."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(self, original_image: Image.Image, original_query: str) -> dict[str, Any]:
        prompt = f"""
You are an expert multimodal task analyst for query-driven synthetic demonstration generation.

Given the attached original_image and the exact original_query below, produce a structured JSON understanding.

CRITICAL INVARIANT:
- Do NOT rewrite, translate, paraphrase, simplify, or replace original_query.
- Every synthetic demonstration will ask exactly the same query string.
- Future synthetic images may change scene/content/layout, but must preserve the task structure required by this exact query.

original_query: {json.dumps(original_query, ensure_ascii=False)}

Analyze:
1. image type and visual domain
2. scene summary
3. target entities / regions / labels referenced by the query
4. attributes or relations being queried
5. likely task family
6. visual evidence needed to answer
7. ambiguity risks if new images are generated

Return ONLY valid JSON with this schema:
{{
  "original_query": string,
  "image_type": string,
  "scene_summary": string,
  "task_family": "comparison|counting|localization|recognition|OCR|spatial_relation|chart_reasoning|document_understanding|UI_understanding|anomaly_detection|visual_reasoning|other",
  "reasoning_type": string,
  "target_entities": [string],
  "reference_entities": [string],
  "queried_attributes": [string],
  "required_visual_evidence": [string],
  "answer_type_hint": string,
  "candidate_answer_space_hint": [string],
  "image_generation_requirements_hint": [string],
  "ambiguity_risks": [string]
}}
""".strip()
        raw = self.backbone.generate_response_multimodal_single(original_image, prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ImageQueryUnderstandingModule expected a JSON object.")
        parsed["original_query"] = original_query
        return parsed

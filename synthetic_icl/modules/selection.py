"""Demonstration selection module."""

from __future__ import annotations

import json
from typing import Any

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import RobustJSONParseError, robust_json_parse
from synthetic_icl.schemas import SyntheticExample


class DemonstrationSelectionModule:
    """Select the most useful synthetic examples for multimodal ICL."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def run(self, candidates: list[SyntheticExample], k: int) -> list[SyntheticExample]:
        if k <= 0 or not candidates:
            return []
        candidate_payload: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            candidate_payload.append(
                {
                    "index": idx,
                    "query": candidate.query,
                    "answer": candidate.answer,
                    "scenario": candidate.scenario.to_dict(),
                    "answer_spec": candidate.answer_spec.to_dict(),
                    "generation_prompt": candidate.generation_prompt.to_dict(),
                    "verification_result": candidate.verification_result,
                    "has_image": candidate.image is not None,
                }
            )

        prompt = f"""
You are selecting synthetic demonstrations for multimodal in-context learning.

Candidates:
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

Select up to {k} candidate indices.

Selection criteria:
- task consistency with the unchanged query
- answer correctness or planned-answer clarity in dry_run
- visual diversity across scenarios/domains
- answer diversity
- low ambiguity
- does not copy original image concrete content
- useful for the original query
- difficulty coverage

Hard constraints:
- Do not modify any candidate query. Each selected example must keep its query unchanged.
- If verification is skipped due to dry_run, rank by scenario diversity, prompt quality, and answer clarity.
- If fewer than {k} are suitable, return fewer.

Return ONLY strict JSON:
{{
  "selected_indices": [0],
  "selection_rationale": string
}}
""".strip()
        raw = self.backbone.generate_response_text(prompt)
        try:
            parsed = robust_json_parse(raw)
            indices = parsed.get("selected_indices", []) if isinstance(parsed, dict) else []
        except (RobustJSONParseError, AttributeError):
            indices = []

        selected: list[SyntheticExample] = []
        seen: set[int] = set()
        for index in indices:
            try:
                idx = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                candidates[idx].selected = True
                selected.append(candidates[idx])
                seen.add(idx)
            if len(selected) >= k:
                break

        if not selected:
            # Conservative fallback keeps dry_run workflows debuggable if selection JSON is malformed.
            selected = candidates[:k]
            for candidate in selected:
                candidate.selected = True
        return selected

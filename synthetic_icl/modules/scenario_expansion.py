"""Scenario expansion module."""

from __future__ import annotations

import json

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import ScenarioSpec, TaskIR


class ScenarioExpansionModule:
    """Generate diverse new scenarios that preserve the same query/task."""

    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    def _is_scenario_aligned(self, task_ir: TaskIR, scenario: ScenarioSpec) -> bool:
        prompt = f"""
You are validating whether a proposed synthetic scenario stays aligned with the source task intent.

Source query:
{json.dumps(task_ir.original_query, ensure_ascii=False)}

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

ScenarioSpec:
{json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2)}

Return ONLY strict JSON:
{{
  "aligned": true,
  "reason": string
}}

aligned=true only if this scenario preserves task type, target comparison/attribute intent, and stays near the original task's visual domain.
""".strip()
        raw = self.backbone.generate_response_text(prompt)
        parsed = robust_json_parse(raw)
        if not isinstance(parsed, dict):
            return False
        return bool(parsed.get("aligned"))

    def run(self, task_ir: TaskIR, num_scenarios: int, max_regen_rounds: int = 3) -> list[ScenarioSpec]:
        prompt = f"""
You are expanding visual scenarios for query-driven synthetic multimodal ICL.

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

Generate {num_scenarios} new ScenarioSpec objects.

Hard constraints:
- The query for every future demonstration MUST be exactly: {json.dumps(task_ir.original_query, ensure_ascii=False)}
- Do NOT create, suggest, or include any new question text.
- Scenarios should not target or copy the original image's concrete content.
- Scenarios may vary in content/layout, but should remain semantically close to the original task domain and preserve the same answerable task structure.
- Each scenario must be directly answerable by the unchanged original_query.
- Prefer domain-near variations: diversify within related visual domains instead of jumping to unrelated domains.
- Cover diverse but related visual domains and difficulty levels.

Return ONLY a strict JSON array. Each object schema:
{{
  "scenario_id": "scenario_001",
  "scenario_description": string,
  "domain": string,
  "how_it_preserves_task": string,
  "how_it_differs_from_original": string,
  "required_objects": [string],
  "required_relations_or_attributes": [string],
  "possible_answers": [string],
  "difficulty_level": "easy|medium|hard"
}}
""".strip()
        if num_scenarios <= 0:
            return []
        aligned_scenarios: list[ScenarioSpec] = []
        seen_signatures: set[str] = set()
        rounds = max(1, int(max_regen_rounds))
        for _ in range(rounds):
            needed = num_scenarios - len(aligned_scenarios)
            if needed <= 0:
                break
            raw = self.backbone.generate_response_text(prompt.replace(f"Generate {num_scenarios} new ScenarioSpec objects.", f"Generate {needed} new ScenarioSpec objects."))
            parsed = robust_json_parse(raw)
            if isinstance(parsed, dict) and "scenarios" in parsed:
                parsed = parsed["scenarios"]
            if not isinstance(parsed, list):
                continue
            scenarios = [ScenarioSpec.from_dict(item) for item in parsed if isinstance(item, dict)]
            for scenario in scenarios:
                if len(aligned_scenarios) >= num_scenarios:
                    break
                signature = f"{scenario.domain}|{scenario.scenario_description}".strip().lower()
                if signature in seen_signatures:
                    continue
                if self._is_scenario_aligned(task_ir, scenario):
                    seen_signatures.add(signature)
                    aligned_scenarios.append(scenario)
        scenarios = aligned_scenarios[:num_scenarios]
        for idx, scenario in enumerate(scenarios, start=1):
            if not scenario.scenario_id:
                scenario.scenario_id = f"scenario_{idx:03d}"
        return scenarios

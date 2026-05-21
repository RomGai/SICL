"""Answer sampling module."""

from __future__ import annotations

import json

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.json_utils import robust_json_parse
from synthetic_icl.schemas import AnswerSpec, ScenarioSpec, TaskIR


class AnswerSamplingModule:
    """Pre-commit answers and image constraints before generation."""

    def __init__(self, backbone: MLLMBackbone, format_retry_times: int = 0) -> None:
        self.backbone = backbone
        self.format_retry_times = max(0, int(format_retry_times))

    def run(
        self,
        task_ir: TaskIR,
        scenarios: list[ScenarioSpec],
        num_answers_per_scenario: int,
        preserve_original_query: bool = True,
    ) -> list[AnswerSpec]:
        query_policy = (
            f"- The query will remain exactly: {json.dumps(task_ir.original_query, ensure_ascii=False)}\n"
            "- Do NOT create, suggest, or include any new question text."
            if preserve_original_query
            else "- For each scenario, create a scenario-matched query that keeps the same task goal and information granularity as TaskIR.original_query."
        )
        prompt = f"""
You are assigning known answers for synthetic multimodal ICL examples BEFORE images are generated.

TaskIR:
{json.dumps(task_ir.to_dict(), ensure_ascii=False, indent=2)}

Scenarios:
{json.dumps([s.to_dict() for s in scenarios], ensure_ascii=False, indent=2)}

For each scenario, create {num_answers_per_scenario} AnswerSpec object(s).

Hard constraints:
{query_policy}
- Do NOT infer answers from an existing generated image; these are planned labels.
- The answer must fit answer_type and candidate_answer_space when available.
- Provide visual constraints that make the known answer unambiguously true.
- Provide negative constraints that prevent ambiguous or conflicting evidence.

Return ONLY a strict JSON array. Each object schema:
{{
  "scenario_id": string,
  "query": string,
  "answer": string,
  "answer_rationale": string,
  "visual_constraints_to_make_answer_true": [string],
  "negative_constraints_to_avoid_ambiguity": [string]
}}
""".strip()
        parsed = None
        last_raw = ""
        max_attempts = self.format_retry_times + 1
        for _ in range(max_attempts):
            raw = self.backbone.generate_response_text(prompt)
            last_raw = raw
            parsed = robust_json_parse(raw)
            if isinstance(parsed, dict) and "answers" in parsed:
                parsed = parsed["answers"]
            if isinstance(parsed, list):
                break

        if not isinstance(parsed, list):
            preview = last_raw[:400].replace("\n", "\\n")
            raise ValueError(
                "AnswerSamplingModule expected a JSON array or {'answers': [...]} "
                f"after {max_attempts} attempt(s). Last raw preview: {preview}"
            )
        scenario_ids = {scenario.scenario_id for scenario in scenarios}
        answers: list[AnswerSpec] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            spec = AnswerSpec.from_dict(item)
            if spec.scenario_id in scenario_ids:
                if preserve_original_query or not spec.query:
                    spec.query = task_ir.original_query
                answers.append(spec)
        return answers

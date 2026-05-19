"""End-to-end pipeline for query-driven synthetic demonstrations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from synthetic_icl.backbone import MLLMBackbone
from synthetic_icl.modules import (
    AnswerSamplingModule,
    DemonstrationSelectionModule,
    GenerationPromptConstructionModule,
    ImageGenerationModule,
    ImageQueryUnderstandingModule,
    ImageRefinementPromptModule,
    ScenarioExpansionModule,
    TaskInductionModule,
    VerificationModule,
)
from synthetic_icl.schemas import SyntheticExample


class SyntheticICLPipeline:
    """Modular orchestration for multimodal synthetic ICL demonstration generation."""

    def __init__(
        self,
        backbone: MLLMBackbone,
        image_understanding_module: ImageQueryUnderstandingModule | None = None,
        task_induction_module: TaskInductionModule | None = None,
        scenario_expansion_module: ScenarioExpansionModule | None = None,
        answer_sampling_module: AnswerSamplingModule | None = None,
        prompt_construction_module: GenerationPromptConstructionModule | None = None,
        image_generation_module: ImageGenerationModule | None = None,
        verification_module: VerificationModule | None = None,
        selection_module: DemonstrationSelectionModule | None = None,
        refinement_prompt_module: ImageRefinementPromptModule | None = None,
    ) -> None:
        self.backbone = backbone
        self.image_understanding_module = image_understanding_module or ImageQueryUnderstandingModule(backbone)
        self.task_induction_module = task_induction_module or TaskInductionModule(backbone)
        self.scenario_expansion_module = scenario_expansion_module or ScenarioExpansionModule(backbone)
        self.answer_sampling_module = answer_sampling_module or AnswerSamplingModule(backbone)
        self.prompt_construction_module = prompt_construction_module or GenerationPromptConstructionModule(backbone)
        self.image_generation_module = image_generation_module or ImageGenerationModule()
        self.verification_module = verification_module or VerificationModule(backbone)
        self.selection_module = selection_module or DemonstrationSelectionModule(backbone)
        self.refinement_prompt_module = refinement_prompt_module or ImageRefinementPromptModule(backbone)
        self.last_candidates: list[SyntheticExample] = []
        self.last_run_log: dict[str, Any] = {}

    @staticmethod
    def _preview(payload: Any, max_chars: int = 1200) -> str:
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except TypeError:
            text = str(payload)
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n...<truncated {len(text) - max_chars} chars>..."

    @staticmethod
    def _log(enabled: bool, stage: str, message: str, payload: Any | None = None) -> None:
        if not enabled:
            return
        print(f"\n[Pipeline:{stage}] {message}")
        if payload is not None:
            print(SyntheticICLPipeline._preview(payload))

    @staticmethod
    def _is_sufficiently_good(verification_result: dict[str, Any]) -> bool:
        if bool(verification_result.get("is_good_enough")):
            return True
        if not bool(verification_result.get("pass")):
            return False
        ambiguity = verification_result.get("ambiguity_score")
        if isinstance(ambiguity, (int, float)):
            return float(ambiguity) <= 0.2
        return False

    @staticmethod
    def _pick_best_attempt(attempt_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not attempt_candidates:
            return None

        def score(item: dict[str, Any]) -> tuple[int, float]:
            result = item.get("verification_result", {})
            passed = 1 if bool(result.get("pass")) else 0
            ambiguity = result.get("ambiguity_score")
            ambiguity_value = float(ambiguity) if isinstance(ambiguity, (int, float)) else 1.0
            return (passed, -ambiguity_value)

        return max(attempt_candidates, key=score)

    def run(
        self,
        original_image: Image.Image,
        original_query: str,
        num_scenarios: int = 5,
        num_answers_per_scenario: int = 1,
        top_k: int = 3,
        dry_run: bool = True,
        verbose: bool = False,
        max_regen_try: int = 3,
        max_edit_try: int = 3,
    ) -> list[SyntheticExample]:
        """Run the full pipeline and return selected synthetic examples.

        dry_run=True still performs all reasoning modules and prompt construction, but it does
        not call the image generation backend; verification is skipped because image=None.

        verbose=True prints stage-by-stage progress and key intermediate results to stdout.
        """
        if original_image is None:
            raise ValueError("original_image must be a PIL image.")
        if not original_query:
            raise ValueError("original_query must be a non-empty string.")

        run_log: dict[str, Any] = {
            "run_config": {
                "original_query": original_query,
                "num_scenarios": num_scenarios,
                "num_answers_per_scenario": num_answers_per_scenario,
                "top_k": top_k,
                "dry_run": dry_run,
                "verbose": verbose,
                "max_regen_try": max_regen_try,
                "max_edit_try": max_edit_try,
            }
        }

        self._log(
            verbose,
            "start",
            "Starting synthetic ICL pipeline run.",
            {
                "original_query": original_query,
                "num_scenarios": num_scenarios,
                "num_answers_per_scenario": num_answers_per_scenario,
                "top_k": top_k,
                "dry_run": dry_run,
            },
        )

        understanding = self.image_understanding_module.run(original_image, original_query)
        run_log["understanding"] = understanding
        self._log(verbose, "understanding", "Image-query understanding completed.", understanding)

        task_ir = self.task_induction_module.run(original_query, understanding)
        run_log["task_ir"] = task_ir.to_dict()
        self._log(verbose, "task_induction", "Task induction completed.", task_ir.to_dict())

        scenarios = self.scenario_expansion_module.run(task_ir, num_scenarios)
        run_log["scenarios"] = [scenario.to_dict() for scenario in scenarios]
        self._log(
            verbose,
            "scenario_expansion",
            f"Scenario expansion completed with {len(scenarios)} scenarios.",
            run_log["scenarios"],
        )

        answer_specs = self.answer_sampling_module.run(task_ir, scenarios, num_answers_per_scenario)
        run_log["answer_specs"] = [answer_spec.to_dict() for answer_spec in answer_specs]
        self._log(
            verbose,
            "answer_sampling",
            f"Answer sampling completed with {len(answer_specs)} answer specs.",
            run_log["answer_specs"],
        )

        scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        candidates: list[SyntheticExample] = []
        candidate_logs: list[dict[str, Any]] = []

        for idx, answer_spec in enumerate(answer_specs, start=1):
            self._log(
                verbose,
                "candidate",
                f"Building candidate {idx}/{len(answer_specs)} for scenario_id={answer_spec.scenario_id}.",
                answer_spec.to_dict(),
            )
            scenario = scenarios_by_id.get(answer_spec.scenario_id)
            if scenario is None:
                self._log(
                    verbose,
                    "candidate",
                    "Skipped candidate because scenario_id was not found in expanded scenarios.",
                    {"scenario_id": answer_spec.scenario_id},
                )
                candidate_logs.append({"scenario_id": answer_spec.scenario_id, "status": "skipped_missing_scenario"})
                continue
            generation_prompt_spec = self.prompt_construction_module.run(
                original_image=original_image,
                task_ir=task_ir,
                scenario=scenario,
                answer_spec=answer_spec,
                original_query=original_query,
            )
            self._log(
                verbose,
                "prompt_construction",
                f"Generation prompt built for scenario_id={scenario.scenario_id}.",
                generation_prompt_spec.to_dict(),
            )

            candidate_log: dict[str, Any] = {
                "scenario_id": scenario.scenario_id,
                "answer_spec": answer_spec.to_dict(),
                "generation_prompt": generation_prompt_spec.to_dict(),
            }

            attempt_candidates: list[dict[str, Any]] = []
            selected_attempt: dict[str, Any] | None = None
            candidate_log["regen_iterations"] = []
            candidate_log["edit_iterations"] = []

            regen_tries = 1 if dry_run else max_regen_try
            for regen_idx in range(regen_tries):
                if dry_run:
                    synthetic_image = None
                    gen_status = {"status": "skipped_dry_run", "regen_try": regen_idx + 1}
                else:
                    try:
                        synthetic_image = self.image_generation_module.generate(original_image, generation_prompt_spec)
                        gen_status = {"status": "completed", "has_image": synthetic_image is not None, "regen_try": regen_idx + 1}
                    except NotImplementedError:
                        synthetic_image = None
                        gen_status = {"status": "not_implemented", "regen_try": regen_idx + 1}

                verification_result = self.verification_module.run(
                    synthetic_image=synthetic_image,
                    original_query=original_query,
                    known_answer=answer_spec.answer,
                    task_ir=task_ir,
                    scenario=scenario,
                    answer_spec=answer_spec,
                )
                attempt = {
                    "image": synthetic_image,
                    "verification_result": verification_result,
                    "stage": "generation",
                    "regen_try": regen_idx + 1,
                }
                attempt_candidates.append(attempt)
                candidate_log["regen_iterations"].append({"generation": gen_status, "verification": verification_result})

                action = str(verification_result.get("recommended_action", "")).lower().strip()
                if action == "accept" or self._is_sufficiently_good(verification_result):
                    selected_attempt = attempt
                    break
                if action == "edit" and bool(verification_result.get("is_valid_demo")) and synthetic_image is not None:
                    selected_attempt = attempt
                    break

                if dry_run or gen_status.get("status") == "not_implemented":
                    break

            if selected_attempt is None:
                # regen failed; skip this scenario/case to avoid noisy examples.
                candidate_log["status"] = "skipped_regen_exhausted"
                candidate_logs.append(candidate_log)
                continue

            current_image = selected_attempt["image"]
            current_verification = selected_attempt["verification_result"]
            current_action = str(current_verification.get("recommended_action", "")).lower().strip()

            if (
                not dry_run
                and current_image is not None
                and current_action == "edit"
                and not self._is_sufficiently_good(current_verification)
            ):
                for edit_idx in range(max_edit_try):
                    edit_prompt = self.refinement_prompt_module.run(
                        original_image=original_image,
                        synthetic_image=current_image,
                        original_query=original_query,
                        task_ir=task_ir,
                        scenario=scenario,
                        answer_spec=answer_spec,
                        base_prompt_spec=generation_prompt_spec,
                        verification_result=current_verification,
                    )
                    edit_prompt_spec = generation_prompt_spec.__class__(
                        scenario_id=generation_prompt_spec.scenario_id,
                        original_query=generation_prompt_spec.original_query,
                        known_answer=generation_prompt_spec.known_answer,
                        image_generation_prompt=edit_prompt,
                        reference_policy=generation_prompt_spec.reference_policy,
                        must_include=generation_prompt_spec.must_include,
                        must_avoid=generation_prompt_spec.must_avoid,
                    )
                    edited_image = self.image_generation_module.generate(current_image, edit_prompt_spec)
                    edited_verification = self.verification_module.run(
                        synthetic_image=edited_image,
                        original_query=original_query,
                        known_answer=answer_spec.answer,
                        task_ir=task_ir,
                        scenario=scenario,
                        answer_spec=answer_spec,
                    )
                    edit_attempt = {
                        "image": edited_image,
                        "verification_result": edited_verification,
                        "stage": "edit",
                        "edit_try": edit_idx + 1,
                    }
                    attempt_candidates.append(edit_attempt)
                    candidate_log["edit_iterations"].append(
                        {"edit_try": edit_idx + 1, "edit_prompt": edit_prompt, "verification": edited_verification}
                    )
                    current_image = edited_image
                    current_verification = edited_verification
                    if self._is_sufficiently_good(edited_verification) or str(edited_verification.get("recommended_action", "")).lower().strip() == "accept":
                        selected_attempt = edit_attempt
                        break

            if not self._is_sufficiently_good(selected_attempt["verification_result"]):
                selected_attempt = self._pick_best_attempt(attempt_candidates)
                if selected_attempt is None:
                    candidate_log["status"] = "skipped_no_valid_attempt"
                    candidate_logs.append(candidate_log)
                    continue

            if selected_attempt is None:
                continue

            synthetic_image = selected_attempt["image"]
            verification_result = selected_attempt["verification_result"]
            candidates.append(
                SyntheticExample(
                    image=synthetic_image,
                    query=original_query,
                    answer=answer_spec.answer,
                    task_ir=task_ir,
                    scenario=scenario,
                    answer_spec=answer_spec,
                    generation_prompt=generation_prompt_spec,
                    verification_result=verification_result,
                    selected=False,
                )
            )
            candidate_logs.append(candidate_log)

        self.last_candidates = candidates
        run_log["candidate_logs"] = candidate_logs
        self._log(verbose, "selection", f"Selecting top_k={top_k} from {len(candidates)} candidates.")
        selected_examples = self.selection_module.run(candidates, top_k)
        run_log["selected_examples"] = [example.to_metadata_dict() for example in selected_examples]
        self._log(
            verbose,
            "done",
            f"Selection completed with {len(selected_examples)} selected examples.",
            run_log["selected_examples"],
        )
        self.last_run_log = run_log
        return selected_examples

"""End-to-end pipeline for query-driven synthetic demonstrations."""

from __future__ import annotations

import json
import random
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
        self.last_attempt_traces: list[dict[str, Any]] = []

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

    def _maybe_randomize_generation_seed(self) -> None:
        if hasattr(self.image_generation_module, "config") and hasattr(self.image_generation_module.config, "seed"):
            self.image_generation_module.config.seed = random.randint(0, 2**31 - 1)

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

        eligible_attempts: list[dict[str, Any]] = []
        for item in attempt_candidates:
            result = item.get("verification_result", {})
            action = str(result.get("recommended_action", "")).lower().strip()
            if not bool(result.get("is_valid_demo")):
                continue
            if action == "regenerate":
                continue
            eligible_attempts.append(item)

        if not eligible_attempts:
            return None

        def score(item: dict[str, Any]) -> tuple[int, float]:
            result = item.get("verification_result", {})
            passed = 1 if bool(result.get("pass")) else 0
            ambiguity = result.get("ambiguity_score")
            ambiguity_value = float(ambiguity) if isinstance(ambiguity, (int, float)) else 1.0
            return (passed, -ambiguity_value)

        return max(eligible_attempts, key=score)

    @staticmethod
    def _history_item(attempt: dict[str, Any]) -> dict[str, Any]:
        result = attempt.get("verification_result", {})
        history: dict[str, Any] = {
            "stage": attempt.get("stage"),
            "recommended_action": result.get("recommended_action"),
            "is_good_enough": result.get("is_good_enough"),
            "is_valid_demo": result.get("is_valid_demo"),
            "failure_type": result.get("failure_type"),
            "ambiguity_score": result.get("ambiguity_score"),
            "issues": result.get("issues", [])[:3],
            "reason": result.get("reason", ""),
            "edit_targets": result.get("edit_targets", [])[:5],
        }
        if "regen_try" in attempt:
            history["regen_try"] = attempt.get("regen_try")
        if "edit_try" in attempt:
            history["edit_try"] = attempt.get("edit_try")
        return history

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
        history_image_window: int = 3,
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
                "history_image_window": history_image_window,
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
        attempt_traces: list[dict[str, Any]] = []

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
            attempt_trace: dict[str, Any] = {
                "scenario_id": scenario.scenario_id,
                "answer_spec": answer_spec.to_dict(),
                "attempts": [],
            }

            attempt_candidates: list[dict[str, Any]] = []
            selected_attempt: dict[str, Any] | None = None
            candidate_log["regen_iterations"] = []
            candidate_log["edit_iterations"] = []

            def recent_history_images(
                max_images: int = history_image_window, *, include_current: bool = True
            ) -> list[Image.Image]:
                imgs = [item.get("image") for item in attempt_candidates if item.get("image") is not None]
                if not include_current and imgs:
                    imgs = imgs[:-1]
                if max_images <= 0:
                    return []
                return imgs[-max_images:]

            regen_tries = 1 if dry_run else max_regen_try
            for regen_idx in range(regen_tries):
                if dry_run:
                    synthetic_image = None
                    gen_status = {"status": "skipped_dry_run", "regen_try": regen_idx + 1}
                else:
                    try:
                        if regen_idx > 0:
                            self._maybe_randomize_generation_seed()
                        synthetic_image = self.image_generation_module.generate(original_image, generation_prompt_spec)
                        gen_status = {"status": "completed", "has_image": synthetic_image is not None, "regen_try": regen_idx + 1}
                    except NotImplementedError:
                        synthetic_image = None
                        gen_status = {"status": "not_implemented", "regen_try": regen_idx + 1}

                history_context = [self._history_item(item) for item in attempt_candidates]
                verification_result = self.verification_module.run(
                    synthetic_image=synthetic_image,
                    original_query=original_query,
                    known_answer=answer_spec.answer,
                    task_ir=task_ir,
                    scenario=scenario,
                    answer_spec=answer_spec,
                    attempt_history=history_context,
                    history_images=recent_history_images(),
                )
                attempt = {
                    "image": synthetic_image,
                    "verification_result": verification_result,
                    "stage": "generation",
                    "regen_try": regen_idx + 1,
                    "prompt_spec": generation_prompt_spec,
                }
                attempt_candidates.append(attempt)
                candidate_log["regen_iterations"].append({"generation": gen_status, "verification": verification_result})
                attempt_trace["attempts"].append(
                    {
                        "stage": "generation",
                        "regen_try": regen_idx + 1,
                        "prompt_spec": generation_prompt_spec.to_dict(),
                        "verification": verification_result,
                        "image": synthetic_image,
                    }
                )

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
                if dry_run and attempt_candidates:
                    # In dry-run, keep prompt-only candidates for inspection even without images.
                    selected_attempt = attempt_candidates[-1]
                    candidate_log["status"] = "selected_dry_run_prompt_only"
                else:
                    # regen failed; skip this scenario/case to avoid noisy examples.
                    candidate_log["status"] = "skipped_regen_exhausted"
                    candidate_logs.append(candidate_log)
                    attempt_trace["status"] = "skipped_regen_exhausted"
                    attempt_traces.append(attempt_trace)
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
                    history_context = [self._history_item(item) for item in attempt_candidates]
                    edit_prompt = self.refinement_prompt_module.run(
                        original_image=original_image,
                        synthetic_image=current_image,
                        original_query=original_query,
                        task_ir=task_ir,
                        scenario=scenario,
                        answer_spec=answer_spec,
                        base_prompt_spec=generation_prompt_spec,
                        verification_result=current_verification,
                        attempt_history=history_context,
                        history_images=recent_history_images(include_current=False),
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
                    self._maybe_randomize_generation_seed()
                    edited_image = self.image_generation_module.generate(current_image, edit_prompt_spec)
                    history_context = [self._history_item(item) for item in attempt_candidates]
                    edited_verification = self.verification_module.run(
                        synthetic_image=edited_image,
                        original_query=original_query,
                        known_answer=answer_spec.answer,
                        task_ir=task_ir,
                        scenario=scenario,
                        answer_spec=answer_spec,
                        attempt_history=history_context,
                        history_images=recent_history_images(),
                    )
                    edit_attempt = {
                        "image": edited_image,
                        "verification_result": edited_verification,
                        "stage": "edit",
                        "edit_try": edit_idx + 1,
                        "prompt_spec": edit_prompt_spec,
                    }
                    attempt_candidates.append(edit_attempt)
                    candidate_log["edit_iterations"].append(
                        {"edit_try": edit_idx + 1, "edit_prompt": edit_prompt, "verification": edited_verification}
                    )
                    attempt_trace["attempts"].append(
                        {
                            "stage": "edit",
                            "edit_try": edit_idx + 1,
                            "edit_prompt": edit_prompt,
                            "prompt_spec": edit_prompt_spec.to_dict(),
                            "verification": edited_verification,
                            "image": edited_image,
                        }
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
                    attempt_trace["status"] = "skipped_no_valid_attempt"
                    attempt_traces.append(attempt_trace)
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
                    generation_prompt=selected_attempt.get("prompt_spec", generation_prompt_spec),
                    verification_result=verification_result,
                    selected=False,
                )
            )
            attempt_trace["status"] = "completed"
            attempt_trace["selected_stage"] = selected_attempt.get("stage")
            attempt_trace["selected_try"] = selected_attempt.get("regen_try", selected_attempt.get("edit_try"))
            attempt_traces.append(attempt_trace)
            candidate_logs.append(candidate_log)

        self.last_candidates = candidates
        self.last_attempt_traces = attempt_traces
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

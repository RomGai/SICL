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
        self.last_candidates: list[SyntheticExample] = []

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

    def run(
        self,
        original_image: Image.Image,
        original_query: str,
        num_scenarios: int = 5,
        num_answers_per_scenario: int = 1,
        top_k: int = 3,
        dry_run: bool = True,
        verbose: bool = False,
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
        self._log(verbose, "understanding", "Image-query understanding completed.", understanding)

        task_ir = self.task_induction_module.run(original_query, understanding)
        self._log(verbose, "task_induction", "Task induction completed.", task_ir.to_dict())

        scenarios = self.scenario_expansion_module.run(task_ir, num_scenarios)
        self._log(
            verbose,
            "scenario_expansion",
            f"Scenario expansion completed with {len(scenarios)} scenarios.",
            [scenario.to_dict() for scenario in scenarios],
        )

        answer_specs = self.answer_sampling_module.run(task_ir, scenarios, num_answers_per_scenario)
        self._log(
            verbose,
            "answer_sampling",
            f"Answer sampling completed with {len(answer_specs)} answer specs.",
            [answer_spec.to_dict() for answer_spec in answer_specs],
        )

        scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        candidates: list[SyntheticExample] = []

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

            if dry_run:
                synthetic_image = None
                self._log(verbose, "image_generation", "Skipped image generation because dry_run=True.")
            else:
                try:
                    synthetic_image = self.image_generation_module.generate(original_image, generation_prompt_spec)
                    self._log(verbose, "image_generation", "Image generation completed for candidate.")
                except NotImplementedError:
                    synthetic_image = None
                    self._log(
                        verbose,
                        "image_generation",
                        "Image generation backend is not implemented; using synthetic_image=None.",
                    )

            verification_result = self.verification_module.run(
                synthetic_image=synthetic_image,
                original_query=original_query,
                known_answer=answer_spec.answer,
                task_ir=task_ir,
                scenario=scenario,
                answer_spec=answer_spec,
            )
            self._log(
                verbose,
                "verification",
                f"Verification completed for scenario_id={scenario.scenario_id}.",
                verification_result,
            )

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

        self.last_candidates = candidates
        self._log(verbose, "selection", f"Selecting top_k={top_k} from {len(candidates)} candidates.")
        selected_examples = self.selection_module.run(candidates, top_k)
        self._log(
            verbose,
            "done",
            f"Selection completed with {len(selected_examples)} selected examples.",
            [example.to_metadata_dict() for example in selected_examples],
        )
        return selected_examples

"""End-to-end pipeline for query-driven synthetic demonstrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    def run(
        self,
        original_image: Image.Image,
        original_query: str,
        num_scenarios: int = 5,
        num_answers_per_scenario: int = 1,
        top_k: int = 3,
        dry_run: bool = True,
    ) -> list[SyntheticExample]:
        """Run the full pipeline and return selected synthetic examples.

        dry_run=True still performs all reasoning modules and prompt construction, but it does
        not call the image generation backend; verification is skipped because image=None.
        """
        if original_image is None:
            raise ValueError("original_image must be a PIL image.")
        if not original_query:
            raise ValueError("original_query must be a non-empty string.")

        understanding = self.image_understanding_module.run(original_image, original_query)
        task_ir = self.task_induction_module.run(original_query, understanding)
        scenarios = self.scenario_expansion_module.run(task_ir, num_scenarios)
        answer_specs = self.answer_sampling_module.run(task_ir, scenarios, num_answers_per_scenario)

        scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        candidates: list[SyntheticExample] = []

        for answer_spec in answer_specs:
            scenario = scenarios_by_id.get(answer_spec.scenario_id)
            if scenario is None:
                continue
            generation_prompt_spec = self.prompt_construction_module.run(
                original_image=original_image,
                task_ir=task_ir,
                scenario=scenario,
                answer_spec=answer_spec,
                original_query=original_query,
            )

            if dry_run:
                synthetic_image = None
            else:
                try:
                    synthetic_image = self.image_generation_module.generate(original_image, generation_prompt_spec)
                except NotImplementedError:
                    synthetic_image = None

            verification_result = self.verification_module.run(
                synthetic_image=synthetic_image,
                original_query=original_query,
                known_answer=answer_spec.answer,
                task_ir=task_ir,
                scenario=scenario,
                answer_spec=answer_spec,
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
        return self.selection_module.run(candidates, top_k)

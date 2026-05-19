"""Core dataclass schemas for query-driven synthetic ICL generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


def _list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


@dataclass
class TaskIR:
    """Task-level intermediate representation induced from an image-query pair."""

    original_query: str
    image_type: str = ""
    scene_summary: str = ""
    task_family: str = "other"
    reasoning_type: str = ""
    target_entities: list[str] = field(default_factory=list)
    reference_entities: list[str] = field(default_factory=list)
    queried_attributes: list[str] = field(default_factory=list)
    required_visual_evidence: list[str] = field(default_factory=list)
    answer_type: str = "free_form"
    candidate_answer_space: list[str] = field(default_factory=list)
    query_invariance_rule: str = "Synthetic examples must use the exact original_query without rewriting."
    image_generation_requirements: list[str] = field(default_factory=list)
    ambiguity_risks: list[str] = field(default_factory=list)
    verification_criteria: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskIR":
        return cls(
            original_query=str(data.get("original_query", "")),
            image_type=str(data.get("image_type", "")),
            scene_summary=str(data.get("scene_summary", "")),
            task_family=str(data.get("task_family", "other")),
            reasoning_type=str(data.get("reasoning_type", "")),
            target_entities=_list_str(data.get("target_entities")),
            reference_entities=_list_str(data.get("reference_entities")),
            queried_attributes=_list_str(data.get("queried_attributes")),
            required_visual_evidence=_list_str(data.get("required_visual_evidence")),
            answer_type=str(data.get("answer_type", "free_form")),
            candidate_answer_space=_list_str(data.get("candidate_answer_space")),
            query_invariance_rule=str(
                data.get(
                    "query_invariance_rule",
                    "Synthetic examples must use the exact original_query without rewriting.",
                )
            ),
            image_generation_requirements=_list_str(data.get("image_generation_requirements")),
            ambiguity_risks=_list_str(data.get("ambiguity_risks")),
            verification_criteria=_list_str(data.get("verification_criteria")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioSpec:
    """A new visual scenario that can still be queried with the unchanged original query."""

    scenario_id: str
    scenario_description: str
    domain: str
    how_it_preserves_task: str
    how_it_differs_from_original: str
    required_objects: list[str] = field(default_factory=list)
    required_relations_or_attributes: list[str] = field(default_factory=list)
    possible_answers: list[str] = field(default_factory=list)
    difficulty_level: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSpec":
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            scenario_description=str(data.get("scenario_description", "")),
            domain=str(data.get("domain", "")),
            how_it_preserves_task=str(data.get("how_it_preserves_task", "")),
            how_it_differs_from_original=str(data.get("how_it_differs_from_original", "")),
            required_objects=_list_str(data.get("required_objects")),
            required_relations_or_attributes=_list_str(data.get("required_relations_or_attributes")),
            possible_answers=_list_str(data.get("possible_answers")),
            difficulty_level=str(data.get("difficulty_level", "medium")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerSpec:
    """A pre-committed answer and visual constraints for one synthetic example."""

    scenario_id: str
    answer: str
    answer_rationale: str
    visual_constraints_to_make_answer_true: list[str] = field(default_factory=list)
    negative_constraints_to_avoid_ambiguity: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerSpec":
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            answer=str(data.get("answer", "")),
            answer_rationale=str(data.get("answer_rationale", "")),
            visual_constraints_to_make_answer_true=_list_str(
                data.get("visual_constraints_to_make_answer_true")
            ),
            negative_constraints_to_avoid_ambiguity=_list_str(
                data.get("negative_constraints_to_avoid_ambiguity")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationPromptSpec:
    """Prompt bundle consumed by a future image generation backend."""

    scenario_id: str
    original_query: str
    known_answer: str
    image_generation_prompt: str
    reference_policy: str
    must_include: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationPromptSpec":
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            original_query=str(data.get("original_query", "")),
            known_answer=str(data.get("known_answer", "")),
            image_generation_prompt=str(data.get("image_generation_prompt", "")),
            reference_policy=str(data.get("reference_policy", "")),
            must_include=_list_str(data.get("must_include")),
            must_avoid=_list_str(data.get("must_avoid")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticExample:
    """One synthetic multimodal ICL demonstration candidate or selected example."""

    image: Image.Image | None
    query: str
    answer: str
    task_ir: TaskIR
    scenario: ScenarioSpec
    answer_spec: AnswerSpec
    generation_prompt: GenerationPromptSpec
    verification_result: dict[str, Any] = field(default_factory=dict)
    selected: bool = False

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize all non-image fields for logging, selection, and demos."""
        return {
            "query": self.query,
            "answer": self.answer,
            "task_ir": self.task_ir.to_dict(),
            "scenario": self.scenario.to_dict(),
            "answer_spec": self.answer_spec.to_dict(),
            "generation_prompt": self.generation_prompt.to_dict(),
            "verification_result": self.verification_result,
            "selected": self.selected,
            "has_image": self.image is not None,
        }


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dictionary with a clear error otherwise."""
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj).__name__}.")
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

"""Query-driven synthetic demonstration generation for multimodal ICL."""

__all__ = [
    "MLLMBackbone",
    "SyntheticICLPipeline",
    "TaskIR",
    "ScenarioSpec",
    "AnswerSpec",
    "GenerationPromptSpec",
    "SyntheticExample",
]


def __getattr__(name: str):
    if name == "MLLMBackbone":
        from synthetic_icl.backbone import MLLMBackbone

        return MLLMBackbone
    if name == "SyntheticICLPipeline":
        from synthetic_icl.pipeline import SyntheticICLPipeline

        return SyntheticICLPipeline
    if name in {"TaskIR", "ScenarioSpec", "AnswerSpec", "GenerationPromptSpec", "SyntheticExample"}:
        from synthetic_icl import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

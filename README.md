# Query-driven Synthetic Demonstration Generation for Multimodal ICL

This repository contains a prototype framework for automatically creating synthetic demonstrations for multimodal in-context learning (ICL).

Given:

- an `original_image`
- an `original_query`

it builds demonstration candidates of the form:

```text
<synthetic_image_i, original_query, known_answer_i>
```

The core design constraint is that the query is invariant: every synthetic example must use the exact same `original_query`. The generated image scenario and pre-committed answer may vary, but the question must not be rewritten.

## Pipeline overview

The default `SyntheticICLPipeline` orchestrates the following modules:

1. **Image-Query Understanding**: uses the MLLM to analyze the original image and query.
2. **Task Induction**: abstracts the concrete image-query pair into a `TaskIR`.
3. **Scenario Expansion**: proposes new visual scenarios that can still be queried with the unchanged original query.
4. **Answer Sampling**: pre-commits known answers and visual constraints before any image is generated.
5. **Generation Prompt Construction**: writes reference-image-conditioned generation prompts.
6. **Image Generation**: currently a stub/placeholder. No real image generation API is called.
7. **Verification**: checks generated images against the unchanged query and known answer. In dry-run mode, this is skipped because there is no generated image.
8. **Demonstration Selection**: ranks candidates for ICL usefulness.

All reasoning modules call the same `MLLMBackbone`. The only exception is `ImageGenerationModule`, which is intentionally isolated so you can replace it with a real generation backend later.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Environment variables

`MLLMBackbone` uses an OpenAI-compatible chat-completions API and reads configuration from environment variables:

```bash
export MLLM_API_KEY="your-api-key"
export MLLM_BASE_URL="https://ai.juguang.chat/v1"
export MLLM_MODEL_NAME="gemini-3-flash-preview-thinking"
```

Defaults:

- `MLLM_BASE_URL=https://ai.juguang.chat/v1`
- `MLLM_MODEL_NAME=gemini-3-flash-preview-thinking`
- `MLLM_API_KEY` is not hardcoded and should be provided by you.

## Running the dry-run demo

The demo reads a local image, accepts the original query, constructs the pipeline, and runs with `dry_run=True`:

```bash
python -m synthetic_icl.demo \
  --image /path/to/original.png \
  --query "子图 A 和 B 谁更加平滑？" \
  --num-scenarios 5 \
  --num-answers-per-scenario 1 \
  --top-k 3
```

It prints:

- `TaskIR`
- `ScenarioSpecs`
- `AnswerSpecs`
- image generation prompts
- selected example metadata

## Dry-run mode

`dry_run=True` is the recommended first step while developing prompts and task schemas.

In dry-run mode:

- the pipeline still calls MLLM reasoning modules;
- it does not call real image generation;
- `SyntheticExample.image` is `None`;
- verification returns a skipped status;
- all metadata, answers, scenarios, and generation prompts are still returned for inspection.

## Replacing the image generation stub

`synthetic_icl/modules/image_generation.py` contains:

```python
class ImageGenerationModule:
    def generate(self, original_image, generation_prompt_spec):
        raise NotImplementedError("Image generation backend is not implemented yet.")
```

To connect a real backend, subclass or replace `ImageGenerationModule` with a class that accepts:

- the original reference image
- a `GenerationPromptSpec`

and returns a `PIL.Image.Image`. The rest of the pipeline does not need to change.

A real backend should follow `GenerationPromptSpec.image_generation_prompt`, especially:

- use the original image only as a task-related style/layout reference;
- do not copy exact original content;
- keep the original query unchanged;
- make the visual evidence clearly support the known answer.

## Why the query must remain unchanged

This framework is query-driven rather than question-generation-driven. The goal is to synthesize demonstrations that teach the model how to answer the user's exact task form for the final original image. If synthetic examples rewrite the query, the ICL context may teach a different instruction pattern and reduce transfer to the final query.

For example, if the original query is:

```text
子图 A 和 B 谁更加平滑？
```

then every synthetic demonstration should still use:

```text
子图 A 和 B 谁更加平滑？
```

The synthetic image can change, such as a new figure with two subplots where A is smoother and B is sharper, and the known answer can be pre-committed as `A`.

## Project structure

```text
synthetic_icl/
  __init__.py
  backbone.py
  schemas.py
  json_utils.py
  modules/
    __init__.py
    understanding.py
    task_induction.py
    scenario_expansion.py
    answer_sampling.py
    prompt_construction.py
    image_generation.py
    verification.py
    selection.py
  pipeline.py
  demo.py
README.md
requirements.txt
```

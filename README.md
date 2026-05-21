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
6. **Image Generation**: defaults to a stub/placeholder, with an optional `qwen_edit` Diffusers backend for Qwen-Image-Edit generation.
7. **Verification**: checks generated images against the unchanged query and known answer. In dry-run mode, this is skipped because there is no generated image.
8. **Demonstration Selection**: ranks candidates for ICL usefulness.

All reasoning modules call the same `MLLMBackbone`. The only exception is `ImageGenerationModule`, which is intentionally isolated so you can replace it with a real generation backend later.

## Installation

Python 3.10+ is recommended.

Base dry-run / MLLM pipeline dependencies:

```bash
pip install -r requirements.txt
```

Optional Qwen image-edit backend dependencies:

```bash
pip install -r requirements-qwen.txt
# equivalent core Diffusers install:
# pip install git+https://github.com/huggingface/diffusers
```

## Config file (recommended)

You can run the demo without `export` by using a single JSON config file. See `synthetic_icl/demo_config.example.json` for a complete example.

```bash
python -m synthetic_icl.demo --config synthetic_icl/demo_config.example.json
```

If `run.log_json_path` (or `--log-json-path`) is set, the demo will save detailed intermediate outputs from each pipeline stage to a JSON file for diagnosis.

Config schema:

- `mllm.api_key`
- `mllm.base_url`
- `mllm.model_name`
- `run.image`
- `run.query`
- `run.num_scenarios`
- `run.num_answers_per_scenario`
- `run.top_k`
- `run.preserve_original_query` (`true` by default; set `false` to allow scenario-matched query rewriting)
- `run.original_image_verify` (`false` by default; set `true` to verify candidates against original-image task/distribution alignment)
- `run.image_generation_pipe` (`stub` or `qwen_edit`)
- `run.dry_run`
- `run.output_dir`
- `run.verbose`
- `run.log_json_path`

CLI flags still work and can override values in the config file.

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

The demo reads a local image, accepts the original query, constructs the pipeline, and runs with `dry_run=True` by default when `--image-generation-pipe stub` is used:

```bash
python -m synthetic_icl.demo \
  --config synthetic_icl/demo_config.example.json
```

## Running optional Qwen image generation

To run real image generation during the image-generation stage, install the optional Qwen dependencies and pass `--image-generation-pipe qwen_edit` at the main entry point:

```bash
python -m synthetic_icl.demo \
  --image /path/to/original.png \
  --query "子图 A 和 B 谁更加平滑？" \
  --image-generation-pipe qwen_edit \
  --output-dir synthetic_outputs \
  --num-scenarios 2 \
  --top-k 2
```

When `--image-generation-pipe qwen_edit` is used, the demo defaults to `dry_run=False`, loads `Qwen/Qwen-Image-Edit-2511`, passes the original image as the reference image list (`image=[original_image]`), and uses each `GenerationPromptSpec.image_generation_prompt` as the Qwen prompt. Generated images are saved under `--output-dir`. You can still force prompt-only execution with `--dry-run`.
For iterative edit runs, the demo also saves process artifacts under `--output-dir/edit_traces/` (per-scenario image trajectory + `trace.json` with verification reasoning and per-round edit prompts). Scenarios filtered as invalid are saved under `--output-dir/invalid_scenarios/` for manual diagnosis.

It prints:

- `TaskIR`
- `ScenarioSpecs`
- `AnswerSpecs`
- image generation prompts
- selected example metadata

You can control how many recent edited images are fed back into verification/refinement context via `history_image_window` (CLI: `--history-image-window`, default `3`).

## Dry-run mode

`dry_run=True` is the recommended first step while developing prompts and task schemas.

In dry-run mode:

- the pipeline still calls MLLM reasoning modules;
- it does not call real image generation;
- `SyntheticExample.image` is `None`;
- verification returns a skipped status;
- all metadata, answers, scenarios, and generation prompts are still returned for inspection.

## Replacing or selecting the image generation backend

`synthetic_icl/modules/image_generation.py` contains both the default stub and an optional `QwenImageEditGenerationModule`. Use the factory to choose a backend programmatically:

```python
from synthetic_icl.modules.image_generation import create_image_generation_module

image_generation_module = create_image_generation_module("qwen_edit")
pipeline = SyntheticICLPipeline(backbone, image_generation_module=image_generation_module)
```

The default stub still raises `NotImplementedError`; this keeps prompt-only development safe and makes image generation opt-in. Any replacement backend should accept:

- the original reference image
- a `GenerationPromptSpec`

and return a `PIL.Image.Image`. The rest of the pipeline does not need to change.

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
requirements-qwen.txt
```

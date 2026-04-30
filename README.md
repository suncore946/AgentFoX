# AgentFoX

Minimal open-source inference code for **AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection**.

This release is scoped to the smallest runnable test path: a user provides images, a CSV with labels, and an LLM/VLM backend. It does not include private API keys, server IPs, absolute paths, image resources, generated profiles, training pipelines, annotation tools, or expert-model assets.

[中文文档](resources/docs/README_ZH.md)

## What Is Included

```text
AgentFoX/
├── agent_pipeline.py
├── forensic_agent/
│   ├── application_builder.py
│   ├── configs/
│   │   ├── config_minimal_test.yaml
│   │   ├── agent_template.txt
│   │   └── prompts/
│   │       ├── reporter_prompt.txt
│   │       └── semantic_analysis.txt
│   ├── core/
│   │   ├── forensic_agent.py
│   │   ├── forensic_llm.py
│   │   ├── forensic_reporter.py
│   │   ├── forensic_tools.py
│   │   └── tools/semantic_analysis_tool.py
│   ├── expert_features/
│   ├── manager/
│   ├── processor/
│   └── utils/
├── docs/README_ZH.md
├── requirements.txt
└── tests/
```

The default workflow runs semantic image analysis and asks the reporter to produce a final binary verdict:

- `0`: authentic camera-captured image
- `1`: AI-generated or forged image

Expert fusion, calibration, clustering profiles, training, labeling, GUI, generated resources, and private datasets are not part of this minimal release.

## Installation

Python 3.10+ is recommended.

```bash
git clone <your-agentfox-repo-url>
cd AgentFoX
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset CSV

Prepare a CSV file such as `test.csv`:

```csv
image_path,gt_label,dataset_name
images/real_001.jpg,0,my_dataset
images/fake_001.png,1,my_dataset
```

Required columns:

- `image_path`: local image path or URL. Relative paths are resolved against `datasets.image_root` when configured; otherwise they are resolved against the CSV file directory.
- `gt_label`: integer label. `0` means authentic image, `1` means AI-generated/forged image.

Optional column:

- `dataset_name`: dataset name for result bookkeeping. If omitted, the CSV filename stem is used.

## LLM Credentials

For OpenAI-compatible backends, set the API key through an environment variable:

```bash
export OPENAI_API_KEY="your_key_here"
```

Do not write API keys into `config_minimal_test.yaml`.

For a local Ollama backend, set:

```yaml
llm:
  model_provider: ollama
  model: qwen2.5vl:7b
  base_url: http://localhost:11434
  support_vision: true
```

## Minimal Run

Edit `forensic_agent/configs/config_minimal_test.yaml` and set:

```yaml
datasets:
  test_paths:
    - /path/to/your/test.csv
```

Then run:

```bash
python agent_pipeline.py \
  --mode test \
  --config_path forensic_agent/configs/config_minimal_test.yaml
```

Single-image analysis:

```bash
python agent_pipeline.py \
  --mode analyze \
  --config_path forensic_agent/configs/config_minimal_test.yaml \
  --image_path /path/to/image.jpg
```

Compute metrics from saved results:

```bash
python agent_pipeline.py \
  --mode only_metrics \
  --config_path forensic_agent/configs/config_minimal_test.yaml
```

Output is written to:

```text
outputs/agentfox_minimal/
├── detail_output/
└── final_output/
```

## Configuration Reference

`agent.*`

- `max_iterations`: maximum LangGraph reasoning iterations.
- `per_workflow_workers`: reserved for future local worker usage; minimal CLI runs serially.
- `open_semantic`: enables the semantic analysis tool.
- `open_expert`, `open_calibration`, `open_clustering`: disabled in the minimal release.
- `agent_template`: path to the Agent system prompt.
- `reporter.prompt_path`: path to the final-report audit prompt.

`llm.*`

- `model_provider`: `openai` for OpenAI-compatible APIs or `ollama` for local Ollama.
- `model`: chat/VLM model name.
- `base_url`: API base URL. Keep private hosts out of committed config files.
- `temperature`, `max_tokens`, `timeout`: generation settings.
- `support_vision`: set `true` when the main model accepts image input.

`datasets.*`

- `test_paths`: one CSV path or a list of CSV paths.
- `image_root`: optional root used to resolve relative `image_path` values.
- `runtime_cache_dir`: optional directory for generated semantic cache; defaults to `.agentfox_cache` beside the first CSV.

`image_manager.*`

- `max_width`, `max_height`: resize bounds for LLM image input.
- `maintain_aspect_ratio`: preserve aspect ratio when resizing.

`logging.*`

- `level`: log level.
- `log_dir`, `file_name`: file log destination.
- `rotation`, `retention`: Loguru file rotation settings.

`tools.SemanticAnalysis.prompt_path`

- Prompt used by the semantic analysis processor.

## Security And Data Boundaries

This repository intentionally does not include:

- API keys or tokens
- private server IPs
- personal absolute paths
- image resources or datasets
- generated semantic/profile/cache/output data
- annotation, model training, or private expert-model pipelines

End-to-end inference requires an external LLM/VLM service. The repository does not run a real networked LLM test unless you provide valid credentials or a local service.

## Verification

```bash
python -m compileall agent_pipeline.py forensic_agent
python agent_pipeline.py --help
pytest
```

Before release, also run a secret/path scan for API-key prefixes, private IPs,
personal absolute paths, and committed API-key fields.

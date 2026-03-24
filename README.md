# AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection

<p align="center">
  <a href="./docs/README_ZH.md">中文</a> |
  <a href="#">arXiv</a> |
  <a href="#">Paper</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/LangGraph-ReAct_Agent-green.svg" />
  <img src="https://img.shields.io/badge/LLM-Qwen3%20%7C%20GPT--4o-orange.svg" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

---

## 📄 Overview

This repository provides the official code implementation and dataset for the paper:

> **AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection**
> *Published on arXiv*

AgentFoX is a Large Language Model–driven forensic framework that redefines AI-Generated Image (AIGI) detection as a **dynamic, multi-phase analytical process**. Instead of returning a coarse binary label, it produces a **detailed, human-readable forensic report** that substantiates its verdict.

---

## 📅 Release Roadmap

We are gradually organizing and releasing resources related to this project:

- [ ] **TODO**: Release the **X-Fuse Dataset**.
- [ ] **TODO**: Release **Verification Code** (evaluation scripts and pre-trained models).
- [ ] **TODO**: Release **Quick Access / Inference Code** for rapid integration.

---

## 📝 Introduction

The increasing realism of AI-Generated Images (AIGI) has created an urgent need for forensic tools capable of reliably distinguishing synthetic content from authentic imagery. Existing detectors are typically tailored to specific forgery artifacts—such as frequency-domain patterns or semantic inconsistencies—leading to specialized performance and, at times, conflicting judgments.

To address these limitations, we present **AgentFoX**, which employs a **quick-integration fusion mechanism** guided by a curated knowledge base comprising:
- 📌 **Calibrated Expert Profiles** — per-model performance profiles with calibrated confidence
- 📌 **Contextual Clustering Profiles** — cluster-aware context for adaptive fusion

During inference, the agent:
1. Begins with **high-level semantic assessment**
2. Transitions to **fine-grained, context-aware synthesis** of signal-level expert evidence
3. Resolves contradictions through **structured reasoning**
4. Outputs a **human-readable forensic report** with an explicit verdict

Beyond detection, this work introduces a **scalable agentic paradigm** that facilitates intelligent integration of future and evolving forensic tools.

---

## 🏗️ System Architecture

```
AgentFoX
├── forensic_agent/
│   ├── core/
│   │   ├── forensic_agent.py        # LangGraph ReAct agent (main orchestrator)
│   │   ├── forensic_llm.py          # Multi-LLM management (Ollama / OpenAI)
│   │   ├── forensic_tools.py        # Auto-discovery tool registration system
│   │   ├── forensic_reporter.py     # Final report generation node
│   │   ├── forensic_template.py     # Agent prompt template builder
│   │   └── agent_state.py           # Stage-based state machine (StageEnum)
│   │
│   ├── core/tools/
│   │   ├── expert_analysis_tool.py  # Expert model result aggregation
│   │   ├── expert_profiles_tool.py  # Expert calibration profile lookup
│   │   ├── clustering_profiles_tool.py  # Cluster-context profile lookup
│   │   └── expert_results_tool.py   # Raw expert prediction retrieval
│   │
│   ├── calibration/                 # Confidence calibration system
│   │   ├── calibration_methods.py   # Temperature scaling, Platt scaling, etc.
│   │   ├── calibration_system.py    # Calibration pipeline
│   │   └── calibration_evaluator.py # ECE, reliability diagram evaluation
│   │
│   ├── processor/                   # Data processing modules
│   │   ├── semantic_forgery_tracking_processor.py
│   │   └── semantic_labeling_processor.py
│   │
│   ├── visualization/               # Result visualization
│   │   ├── report_generator.py
│   │   ├── heatmap_generator.py
│   │   └── performance_visualizer.py
│   │
│   ├── manager/                     # Service managers
│   │   ├── config_manager.py
│   │   ├── datasets_manager.py
│   │   ├── image_manager.py
│   │   ├── profile_manager.py
│   │   └── feature_manager.py
│   │
│   └── configs/
│       ├── profiles/                # Expert & clustering profiles (JSON)
│       └── prompts/                 # All agent/tool prompts (TXT)
│
├── agent_pipeline.py                # Main entry point (batch test / single analyze)
├── app_gui.py                       # Gradio web UI
├── calibration_profile_pipeline.py  # Build calibration profiles
├── clustering_profile_pipeline.py   # Build clustering profiles
└── model_profile_pipeline.py        # Build model performance profiles
```

---

## 🔍 Multi-Stage Analysis Workflow

AgentFoX follows a **stage-based reasoning pipeline** driven by a ReAct agent:

| Stage | Description |
|-------|-------------|
| `INITIAL` | Image ingestion and setup |
| `SEMANTIC_LEVEL` | High-level semantic plausibility check (VLM-based) |
| `EXPERT_PROFILES` | Retrieve calibrated expert model performance profiles |
| `EXPERT_RESULTS` | Collect raw predictions from all expert detectors |
| `EXPERT_ANALYSIS` | Fuse and analyze expert evidence with cluster context |
| `CLUSTERING_ANALYSIS` | Apply clustering profiles to resolve conflicting signals |
| `FINALLY_REPORT` | Generate structured forensic report with final verdict |

The agent uses `update stage to: <stage_name>` as a lightweight state-transition protocol, making the pipeline auditable and explainable.

---

## 🧰 Expert Detectors Supported

AgentFoX integrates multiple AIGI detection expert models as tools:

| Model | Focus Area |
|-------|-----------|
| **DRCT** | Frequency & texture artifacts |
| **RINE** | Noise inconsistency detection |
| **SPAI** | Semantic-physical anomaly identification |
| **Patch Shuffle** | Patch-level statistical analysis |

Expert outputs are **probability-calibrated** via temperature scaling / Platt scaling before being passed to the agent.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/your-repo/AgentFoX.git
cd AgentFoX
pip install -r requirements.txt
```

**Key dependencies:**

| Category | Libraries |
|----------|-----------|
| LLM / Agent | `langchain`, `langgraph`, `langchain-openai` |
| Vision | `Pillow`, `opencv-python`, `scikit-image` |
| ML | `torch`, `torchvision`, `timm`, `scikit-learn` |
| Data | `pandas`, `numpy`, `faiss-cpu` |
| UI | `gradio` |
| Logging | `loguru`, `tqdm` |

### 2. Configuration

Edit or create a YAML config (see `forensic_agent/configs/config_qwen3_32b_benchmark.yaml`):

```yaml
# Agent reasoning engine
agent:
  max_iterations: 40
  open_calibration: true   # Enable calibration profiles
  open_semantic: true      # Enable semantic analysis
  open_expert: true        # Enable expert model analysis
  open_clustering: true    # Enable clustering profiles
  expert_models:
    - "DRCT"
    - "PatchShuffle"
    - "SPAI"
    - "RINE"

# LLM backbone (Ollama local or OpenAI-compatible API)
llm:
  model: "qwen3:32b"
  base_url:
    - "http://localhost:11434"
  model_provider: "Ollama"
  temperature: 0
  reasoning: true
```

### 3. Run Batch Evaluation

```bash
# Batch test on a dataset
python agent_pipeline.py \
    --mode test \
    --config_path forensic_agent/configs/config_qwen3_32b_benchmark.yaml

# Compute metrics from results
python agent_pipeline.py \
    --mode metrics \
    --config_path forensic_agent/configs/config_qwen3_32b_benchmark.yaml

# Analyze a single image
python agent_pipeline.py \
    --mode analyze \
    --config_path forensic_agent/configs/config_qwen3_32b_benchmark.yaml \
    --image_path /path/to/image.jpg

# Extract semantic features
python agent_pipeline.py \
    --mode semantic \
    --config_path forensic_agent/configs/config_qwen3_32b_benchmark.yaml
```

### 4. Launch Web UI (Gradio)

```bash
python app_gui.py
# Access at: http://0.0.0.0:7860
```

The web UI provides:
- 🖼️ **Image upload** or gallery example selection
- 🔴/🟢 **Visual verdict** (FAKE / REAL)
- 🔗 **Process graph** — node-by-node agent reasoning visualization
- 📝 **Forensic report** — human-readable analysis
- 💬 **Chat detail** — full agent conversation trace
- 📄 **Raw JSON** — structured output for downstream use

### 5. Build Knowledge Base Profiles

Before running the agent, pre-build the required profile files:

```bash
# Step 1: Build calibration profiles for expert models
python calibration_profile_pipeline.py

# Step 2: Build clustering profiles
python clustering_profile_pipeline.py

# Step 3: Build expert model performance profiles
python model_profile_pipeline.py
```

---

## ⚙️ Run Modes

| Mode | Command Flag | Description |
|------|-------------|-------------|
| Batch Test | `--mode test` | Run AgentFoX on all images in the configured dataset |
| Metrics | `--mode metrics` | Compute ACC / F1 / Precision / Recall from saved results |
| Single Image | `--mode analyze` | Analyze one image and print the forensic report |
| Semantic Extract | `--mode semantic` | Pre-extract semantic features (VLM-based) for all images |
| Expert Extract | `--mode expert` | Pre-extract expert model predictions for all images |

---

## 📂 X-Fuse Dataset

The **X-Fuse** dataset is specifically designed for evaluating **explainable** AI-generated image detection, covering a rich variety of samples from multiple generative models.

- **Download Link**: Coming soon (TODO)
- **Data Structure**: Details will be updated upon release

---

## 📊 Supported Datasets for Evaluation

| Dataset | Description |
|---------|-------------|
| GenImage | Large-scale AIGI benchmark |
| GenImage-Val | Validation split |
| AIGCDetect-testset | Multi-source AI detection |
| AIGIBench | Comprehensive AIGI benchmark |
| Chameleon | Challenging adversarial samples |
| Community-Forensics | Real-world community media |
| WildRF / WIRA | In-the-wild realistic forgeries |

---

## 📧 Contact

If you have any questions regarding the code or dataset, please feel free to open an Issue or contact us via email:

📮 **2453043007@mails.szu.edu.cn**

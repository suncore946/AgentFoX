# AgentFoX 最小开源版

本仓库提供 **AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection** 的最小可用推理代码。

本次开源范围只覆盖“用户有图片和标签 CSV，即可跑通 `agent_pipeline.py --mode test`”这一条路径。仓库不包含 API key、私有服务器 IP、个人绝对路径、图片资源、生成 profile、训练流水线、标注工具或专家模型资产。

[English README](../README.md)

## 目录结构

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

默认流程只启用语义分析工具，并输出二分类结论：

- `0`：真实相机图像
- `1`：AI 生成或伪造图像

完整专家融合、校准、聚类 profile、训练、标注、GUI、生成资源和私有数据集不属于本次最小开源目标。

## 环境安装

推荐 Python 3.10+。

```bash
git clone <your-agentfox-repo-url>
cd AgentFoX
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 待测数据集组成

准备一个 `test.csv`：

```csv
image_path,gt_label,dataset_name
images/real_001.jpg,0,my_dataset
images/fake_001.png,1,my_dataset
```

必需列：

- `image_path`：本地图片路径或 URL。相对路径优先按 `datasets.image_root` 解析；未配置时按 CSV 所在目录解析。
- `gt_label`：整数标签。`0=真实图像`，`1=AI生成图像/伪造图像`。

可选列：

- `dataset_name`：数据集名称，用于结果记录；缺省时使用 CSV 文件名。

## LLM 凭据

OpenAI-compatible 后端通过环境变量提供 API key：

```bash
export OPENAI_API_KEY="your_key_here"
```

不要把 API key 写入 `config_minimal_test.yaml`。

如果使用本地 Ollama，可在配置中改为：

```yaml
llm:
  model_provider: ollama
  model: qwen2.5vl:7b
  base_url: http://localhost:11434
  support_vision: true
```

## 最小执行方法

编辑 `forensic_agent/configs/config_minimal_test.yaml`：

```yaml
datasets:
  test_paths:
    - /path/to/your/test.csv
```

运行批量 test：

```bash
python agent_pipeline.py \
  --mode test \
  --config_path forensic_agent/configs/config_minimal_test.yaml
```

运行单图分析：

```bash
python agent_pipeline.py \
  --mode analyze \
  --config_path forensic_agent/configs/config_minimal_test.yaml \
  --image_path /path/to/image.jpg
```

统计已保存结果：

```bash
python agent_pipeline.py \
  --mode only_metrics \
  --config_path forensic_agent/configs/config_minimal_test.yaml
```

输出目录：

```text
outputs/agentfox_minimal/
├── detail_output/
└── final_output/
```

## 配置条目说明

`agent.*`

- `max_iterations`：LangGraph 推理最大轮数。
- `per_workflow_workers`：预留给未来本地 worker；当前最小 CLI 串行运行。
- `open_semantic`：启用语义分析工具。
- `open_expert`、`open_calibration`、`open_clustering`：最小版默认关闭。
- `agent_template`：Agent 主提示词路径。
- `reporter.prompt_path`：最终报告审计提示词路径。

`llm.*`

- `model_provider`：`openai` 表示 OpenAI-compatible API，`ollama` 表示本地 Ollama。
- `model`：聊天/VLM 模型名。
- `base_url`：API 地址。不要把私有地址提交到公开仓库。
- `temperature`、`max_tokens`、`timeout`：生成参数。
- `support_vision`：主模型是否支持图片输入。

`datasets.*`

- `test_paths`：一个 CSV 路径或多个 CSV 路径。
- `image_root`：可选，用于解析 CSV 中的相对 `image_path`。
- `runtime_cache_dir`：可选，语义分析运行时缓存目录；默认在第一个 CSV 同级 `.agentfox_cache`。

`image_manager.*`

- `max_width`、`max_height`：发送给 LLM 的图片缩放上限。
- `maintain_aspect_ratio`：缩放时是否保持长宽比。

`logging.*`

- `level`：日志级别。
- `log_dir`、`file_name`：日志文件位置。
- `rotation`、`retention`：Loguru 日志轮转配置。

`tools.SemanticAnalysis.prompt_path`

- 语义分析处理器使用的提示词路径。

## 安全边界

仓库刻意不包含：

- API key 或 token
- 私有服务器 IP
- 个人绝对路径
- 图像资源或数据集
- 运行生成的 semantic/profile/cache/output 数据
- 标注、模型训练或私有专家模型流水线

真实端到端推理需要外部 LLM/VLM 服务。除非你提供有效 key 或本地 VLM 服务，否则本仓库不会跑真实联网 LLM 测试。

## 验证命令

```bash
python -m compileall agent_pipeline.py forensic_agent
python agent_pipeline.py --help
pytest
```

发布前还应扫描 API-key 前缀、私有 IP、个人绝对路径和已提交的 API-key 字段。

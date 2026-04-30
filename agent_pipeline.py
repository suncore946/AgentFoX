"""AgentFoX minimal inference entrypoint.

中文说明: 这个文件只保留开源版最小推理流程, 用于批量执行 test 任务和单图 analyze。
English: This file keeps the minimal open-source inference flow for batch test
and single-image analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from forensic_agent.core.forensic_dataclass import ProcessingResult
from forensic_agent.manager.datasets_manager import DatasetsManager


class AFAApplication:
    """AgentFoX runtime application.

    中文说明: 负责读取配置、初始化依赖容器、执行图片分析并保存结果。
    English: Loads configuration, initializes the dependency container, runs
    image analysis, and writes results.
    """

    def __init__(self, config_path: str | Path | None = None, is_debug: bool = False, max_workers: int = 1, **_: Any) -> None:
        if config_path is None:
            raise ValueError("config_path is required for AgentFoX inference.")

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.is_debug = is_debug
        self.max_workers = max(1, int(max_workers or 1))
        self._container = None
        self._initialized = False
        self.logger = logger

        save_dir = self.config.get("target_dir") or self.config.get("agent", {}).get("save_dir", "./outputs/agent_results")
        self.save_dir = Path(save_dir)
        self.detail_save_dir = self.save_dir / "detail_output"
        self.finally_save_dir = self.save_dir / "final_output"
        self.detail_save_dir.mkdir(parents=True, exist_ok=True)
        self.finally_save_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Initialize all runtime services.

        中文说明: 延迟初始化可让 `python agent_pipeline.py --help` 不触发 LLM 连接。
        English: Lazy initialization keeps `python agent_pipeline.py --help`
        from connecting to an LLM service.
        """
        if self._initialized:
            return

        from forensic_agent.application_builder import ApplicationBuilder
        from forensic_agent.core.forensic_agent import ForensicAgent
        from forensic_agent.manager.config_manager import ConfigManager
        from forensic_agent.manager.image_manager import ImageManager
        from forensic_agent.manager.logger_manager import LoggerManager

        self._container = ApplicationBuilder(self.config_path, self.is_debug).build()
        self.config_manager: ConfigManager = self._container.get_service(ConfigManager)
        self.logger: LoggerManager = self._container.get_service(LoggerManager)
        self.image_manager: ImageManager = self._container.get_service(ImageManager)
        self.datasets_manager: DatasetsManager = self._container.get_service(DatasetsManager)
        self.agent: ForensicAgent = self._container.get_service(ForensicAgent)
        self._initialized = True
        self.logger.info("AgentFoX application initialized.")

    @staticmethod
    def get_file_name(image_path: str | Path) -> str:
        """Return a stable result filename for an image path.

        中文说明: 结果文件名使用路径 MD5, 避免不同目录下同名图片互相覆盖。
        English: The result filename uses an MD5 hash of the path so images
        with the same basename from different folders do not collide.
        """
        name = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()
        return f"{name}.json"

    @staticmethod
    def _redact_message_dump(message_dump: dict[str, Any]) -> dict[str, Any]:
        """Remove embedded image base64 before saving detail output.

        中文说明: detail_output 只保存推理文本和工具结果, 不落盘用户图片内容。
        English: detail_output stores reasoning text and tool results only, not
        user image content.
        """

        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                redacted = {}
                for key, item in value.items():
                    if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
                        redacted[key] = "<redacted image data>"
                    else:
                        redacted[key] = redact(item)
                return redacted
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        return redact(message_dump)

    def analyze_single_image(self, image_path: str | Path) -> ProcessingResult:
        """Analyze one image with the configured AgentFoX workflow.

        中文说明: 单图模式复用 test 模式同一条 Agent 工作流。
        English: Single-image mode reuses the same agent workflow as batch
        test mode.
        """
        if not self._initialized:
            self.initialize()
        normalized_path = DatasetsManager.normalize_image_path(image_path)
        return self.agent.think_and_act(normalized_path, workflow_id=0)

    def test(self) -> None:
        """Run batch inference over the configured test CSV.

        中文说明: 输入 CSV 只要求包含 image_path 和 gt_label, dataset_name 可选。
        English: The input CSV only requires image_path and gt_label;
        dataset_name is optional.
        """
        if not self._initialized:
            self.initialize()

        detail_data = self.datasets_manager.detail_data.sort_values(by="image_path")
        grouped_image_info = detail_data.groupby("image_path", sort=True)

        for image_path, group_data in tqdm(grouped_image_info, total=len(grouped_image_info), desc="AgentFoX test"):
            result_path = self.finally_save_dir / self.get_file_name(image_path)
            if result_path.exists():
                continue
            try:
                analysis_result = self.agent.think_and_act(image_path, workflow_id=0)
                self._save_result(image_path, group_data, analysis_result)
            except Exception as exc:
                self.logger.exception(f"Failed to process image {image_path}: {exc}")

    def _save_result(self, image_path: str, group_data: pd.DataFrame, analysis_result: dict) -> None:
        """Persist detailed messages and final structured output.

        中文说明: detail_output 保存完整消息链, final_output 保存最终结构化判断。
        English: detail_output stores the full message chain, while
        final_output stores the final structured verdict.
        """
        final_response = analysis_result.get("final_response")
        if not final_response:
            self.logger.error(f"Image {image_path} has no final_response; skipping save.")
            return
        if final_response.get("is_success") is False:
            self.logger.error(f"Image {image_path} produced an unsuccessful final_response: {final_response}")
            return

        final_response["image_path"] = image_path
        final_response["gt_label"] = int(group_data["gt_label"].iloc[0])
        final_response["dataset_name"] = (
            str(group_data["dataset_name"].iloc[0]) if "dataset_name" in group_data.columns else "default"
        )
        final_response["metrics"] = analysis_result.get("metrics", final_response.get("metrics", {}))

        save_file_name = self.get_file_name(image_path)
        with open(self.detail_save_dir / save_file_name, "w", encoding="utf-8") as f:
            messages = [
                self._redact_message_dump(m.model_dump())
                for m in analysis_result.get("messages", [])
                if getattr(m, "content", None)
            ]
            json.dump(messages, f, ensure_ascii=False, indent=2)

        with open(self.finally_save_dir / save_file_name, "w", encoding="utf-8") as f:
            json.dump(final_response, f, ensure_ascii=False, indent=2)

    def metrics_for_dir(self, target_dir: str | Path | None = None) -> None:
        """Compute simple metrics from saved final_output JSON files.

        中文说明: 这是一个轻量验收工具, 只统计 LLM 最终判断与 gt_label。
        English: This lightweight acceptance helper only compares the LLM final
        verdict with gt_label.
        """
        target = Path(target_dir) if target_dir else self.finally_save_dir
        if target.name != "final_output":
            target = target / "final_output"

        rows = []
        for json_file in sorted(target.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pred = data.get("finally_result")
            gt = data.get("gt_label")
            if pred in {0, 1} and gt in {0, 1}:
                rows.append({"gt_label": int(gt), "pred_label": int(pred)})

        if not rows:
            self.logger.warning(f"No valid result JSON files found under {target}.")
            return

        df = pd.DataFrame(rows)
        summary = {
            "Sample_Count": len(df),
            "Accuracy": round(accuracy_score(df["gt_label"], df["pred_label"]), 4),
            "F1": round(f1_score(df["gt_label"], df["pred_label"], zero_division=0), 4),
            "Precision": round(precision_score(df["gt_label"], df["pred_label"], zero_division=0), 4),
            "Recall": round(recall_score(df["gt_label"], df["pred_label"], zero_division=0), 4),
        }
        output_file = target.parent / "prediction_metrics.csv"
        pd.DataFrame([summary]).to_csv(output_file, index=False)
        self.logger.info(f"Metrics saved to {output_file}")


def main(args: argparse.Namespace) -> None:
    """CLI dispatcher.

    中文说明: CLI 只暴露最小开源版需要的 test/analyze/only_metrics。
    English: The CLI exposes only the minimal open-source test/analyze/
    only_metrics modes.
    """
    app = AFAApplication(**vars(args))

    if args.mode == "test":
        app.test()
    elif args.mode == "analyze":
        if not args.image_path:
            raise ValueError("--image_path is required in analyze mode.")
        result = app.analyze_single_image(args.image_path)
        print(json.dumps(result.get("final_response", result), ensure_ascii=False, indent=2))
    elif args.mode == "only_metrics":
        app.metrics_for_dir(args.metrics_dir)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentFoX minimal image forensics inference")
    parser.add_argument("--config_path", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--mode", choices=["test", "analyze", "only_metrics"], default="test")
    parser.add_argument("--image_path", type=str, default=None, help="Image path for analyze mode.")
    parser.add_argument("--metrics_dir", type=str, default=None, help="Result directory for only_metrics mode.")
    parser.add_argument("--is_debug", action="store_true", default=False, help="Enable LangGraph/LangChain debug output.")
    parser.add_argument("--max_workers", type=int, default=1, help="Reserved for future local parallelism.")
    main(parser.parse_args())

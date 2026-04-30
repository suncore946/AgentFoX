"""Minimal runtime tests for AgentFoX.

中文说明: 这些测试不调用真实 LLM, 只验证 CSV 解析、路径解析和无专家工具容错。
English: These tests do not call a real LLM; they validate CSV parsing, path
resolution, and behavior without expert tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.messages import ToolMessage

from forensic_agent.core.forensic_reporter import ForensicReporter
from forensic_agent.manager.datasets_manager import DatasetsManager


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a small test CSV.

    中文说明: 使用 pandas 写入以保持和运行时读取逻辑一致。
    English: Uses pandas so test CSV output matches runtime loading behavior.
    """
    pd.DataFrame(rows).to_csv(path, index=False)


def test_minimal_csv_loads_and_groups_by_image_path(tmp_path: Path) -> None:
    """Minimal image_path/gt_label CSV can be loaded and grouped.

    中文说明: dataset_name 缺省时自动使用 CSV 文件名。
    English: When dataset_name is missing, the CSV stem is used automatically.
    """
    csv_path = tmp_path / "test.csv"
    write_csv(csv_path, [{"image_path": "images/a.jpg", "gt_label": 1}])

    manager = DatasetsManager({"test_paths": [str(csv_path)]})
    data = manager.detail_data

    expected_path = (tmp_path / "images/a.jpg").resolve(strict=False).as_posix()
    assert data.loc[0, "image_path"] == expected_path
    assert data.loc[0, "gt_label"] == 1
    assert data.loc[0, "dataset_name"] == "test"
    assert len(list(data.groupby("image_path"))) == 1


def test_relative_image_path_uses_image_root_when_configured(tmp_path: Path) -> None:
    """datasets.image_root overrides CSV parent for relative paths.

    中文说明: 用户把图片集中放到单独目录时可用 image_root。
    English: image_root supports users who keep images in a separate directory.
    """
    csv_path = tmp_path / "labels" / "test.csv"
    csv_path.parent.mkdir()
    image_root = tmp_path / "images_root"
    write_csv(csv_path, [{"image_path": "nested/a.png", "gt_label": 0, "dataset_name": "custom"}])

    manager = DatasetsManager({"test_paths": str(csv_path), "image_root": str(image_root)})
    data = manager.detail_data

    expected_path = (image_root / "nested/a.png").resolve(strict=False).as_posix()
    assert data.loc[0, "image_path"] == expected_path
    assert data.loc[0, "dataset_name"] == "custom"


def test_reporter_missing_expert_tool_result_returns_empty_dict() -> None:
    """Reporter tolerates absent expert tool messages.

    中文说明: 最小配置没有 expert_results 工具, 因此缺失时必须返回空字典。
    English: The minimal config has no expert_results tool, so missing output
    must become an empty dict.
    """
    reporter = ForensicReporter.__new__(ForensicReporter)
    state = {
        "messages": [
            ToolMessage(
                content=json.dumps({"semantic_result": 1}),
                name="semantic_analysis",
                tool_call_id="semantic-call",
            )
        ]
    }

    assert reporter._load_tool_result(state, "semantic_analysis") == {"semantic_result": 1}
    assert reporter._load_tool_result(state, "expert_results") == {}

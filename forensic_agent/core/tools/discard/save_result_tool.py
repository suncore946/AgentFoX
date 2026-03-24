import json
from typing import Dict, Any
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

from ..tools_base import ToolsBase


class SaveJsonToolSchema(BaseModel):
    """save_json工具的输入结构"""

    image_path: str = Field(..., description="请提供图像名称")
    save_data: Any = Field(..., description="要保存为JSON的数据（必须是可JSON序列化的字典）")

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """自定义验证：确保data可JSON序列化"""
        try:
            json.dumps(value)
        except TypeError as e:
            raise ValueError(f"data不可JSON序列化: {e}")
        return value


class SaveJsonTool(ToolsBase):
    """将输入内容保存为JSON文件的工具"""

    # 将改进后的schema作为类属性，便于ForensicTools注入到ToolsAdapter
    args_schema = SaveJsonToolSchema

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.save_dir = Path(config.get("save_dir", "outputs/agent_reports"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "save_result"

    @property
    def description(self) -> str:
        return "将输出结果保存为JSON文件"

    def execute(self, **kwargs: Any):
        """保存数据为JSON文件"""
        # 使用schema验证和解析输入（确保类型安全）
        params: SaveJsonToolSchema = self.args_schema.validate_data(kwargs)
        final_save_path = self.save_dir / f"{Path(params.image_path).stem}.json"
        logger.info(f"开始保存JSON - 路径: {final_save_path}, 数据键: {list(params.data.keys())}")
        with open(final_save_path, "w", encoding="utf-8") as f:
            json.dump(params.save_data, f, ensure_ascii=False, indent=4)
        return f"数据已成功保存到 {final_save_path}"

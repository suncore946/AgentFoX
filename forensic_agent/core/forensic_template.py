# forensic_template.py
from pathlib import Path
from typing import Dict, Any, List
from string import Formatter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from loguru import logger
from .core_exceptions import ForensicTemplateError


class ForensicTemplate:
    """
    取证提示模板
    """

    # 类常量定义
    DEFAULT_AGENT_TEMPLATE_FILE = "agent_template.txt"

    # 配置键常量
    CONFIG_AGENT_TEMPLATE = "agent_template"

    def __init__(self, agent_config: Dict[str, Any]):
        """初始化模板管理器"""
        self.agent_config = agent_config

        # 设置模板路径
        self.agent_template_path = self._get_template_path(self.CONFIG_AGENT_TEMPLATE, self.DEFAULT_AGENT_TEMPLATE_FILE)

        # 加载模板内容
        self._agent_template = self._load_template(self.agent_template_path, "Agent模板")

        logger.info("ForensicTemplate初始化完成")

    def _get_template_path(self, config_key: str, default_filename: str) -> Path:
        """获取模板文件路径"""
        custom_path = self.agent_config.get(config_key)
        if custom_path:
            return Path(custom_path)
        else:
            return Path(__file__).parent.parent / "configs" / "prompts" / default_filename

    def _load_template(self, template_path: Path, template_name: str) -> str:
        """加载模板内容"""
        if template_path.exists():
            return self._read_template_file(template_path, template_name)
        else:
            logger.error(f"{template_name}文件不存在: {template_path}")
            raise ForensicTemplateError(f"{template_name}文件不存在: {template_path}")

    def _read_template_file(self, template_path: Path, template_name: str) -> str:
        """读取模板文件内容"""
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ForensicTemplateError(f"{template_name}文件为空")
                logger.debug(f"成功加载{template_name}: {template_path}")
                return content
        except UnicodeDecodeError as e:
            raise ForensicTemplateError(f"{template_name}文件编码错误: {e}") from e
        except OSError as e:
            raise ForensicTemplateError(f"读取{template_name}文件失败: {e}") from e

    def build_agent_template(self, exclude_variables: List[str] = ["tools", "tool_names"]) -> PromptTemplate:
        """创建LangChain Agent提示模板"""
        try:
            all_variables = self._extract_template_variables(self._agent_template)
            input_variables = [v for v in all_variables if v not in exclude_variables]
            logger.debug(f"提取的模板变量: {input_variables}")
            return PromptTemplate(template=self._agent_template, input_variables=input_variables)
        except Exception as e:
            logger.error(f"创建Agent模板失败: {e}")
            raise ForensicTemplateError(f"创建Agent模板失败: {e}") from e

    def _extract_template_variables(self, template_str: str) -> List[str]:
        """从模板字符串中提取变量名"""
        if not template_str:
            return []

        formatter = Formatter()
        variables = []

        try:
            for _, field_name, _, _ in formatter.parse(template_str):
                if field_name is not None and field_name:
                    # 处理复杂的字段名，如 obj.attr 或 arr[0]
                    base_name = field_name.split(".")[0].split("[")[0]
                    if base_name and base_name not in variables:
                        variables.append(base_name)
        except Exception as e:
            logger.warning(f"解析模板变量失败: {e}")
            # 返回常用变量作为后备
            return ["input", "tools", "current_stage", "messages", "agent_scratchpad"]

        return variables

    def format_system_message(self, goal: str) -> SystemMessage:
        """格式化系统消息"""
        try:
            formatted_content = self._system_template.format(goal=goal.strip())
            return SystemMessage(content=formatted_content)
        except Exception as e:
            logger.error(f"格式化系统消息失败: {e}")
            return SystemMessage(content=f"你是一个专业的AIGC图像取证分析师。任务: {goal}")

    @property
    def agent_template(self) -> str:
        """获取Agent模板内容"""
        return self._agent_template

    @property
    def system_template(self) -> str:
        """获取系统模板内容"""
        return self._system_template

    def reload_templates(self) -> None:
        """重新加载模板内容"""
        self._agent_template = self._load_template(self.agent_template_path, self.DEFAULT_AGENT_TEMPLATE, "Agent模板")
        self._system_template = self._load_template(self.system_template_path, self.DEFAULT_SYSTEM_TEMPLATE, "系统模板")
        logger.info("已重新加载所有模板")

    def get_template_info(self) -> Dict[str, Any]:
        """获取模板信息"""
        return {
            "agent_template_path": str(self.agent_template_path),
            "system_template_path": str(self.system_template_path),
            "agent_template_exists": self.agent_template_path.exists(),
            "system_template_exists": self.system_template_path.exists(),
            "agent_template_loaded": bool(self._agent_template),
            "system_template_loaded": bool(self._system_template),
            "agent_variables": self._extract_template_variables(self._agent_template),
        }

"""Prompt template loader for AgentFoX.

中文说明: 这里只加载 Agent 主提示模板, 不再依赖私有实验系统提示。
English: This module loads only the main agent prompt template and no longer
depends on private experimental system prompts.
"""

from pathlib import Path
from typing import Dict, Any, List
from string import Formatter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from loguru import logger
from .core_exceptions import ForensicTemplateError


class ForensicTemplate:
    """Forensic prompt template helper.

    中文说明: agent.agent_template 可覆盖默认模板路径, 方便用户自定义最小流程提示。
    English: agent.agent_template can override the default template path so
    users can customize the minimal workflow prompt.
    """

    DEFAULT_AGENT_TEMPLATE_FILE = "agent_template.txt"

    CONFIG_AGENT_TEMPLATE = "agent_template"

    def __init__(self, agent_config: Dict[str, Any]):
        """Initialize template manager.

        中文说明: 配置缺省时读取 forensic_agent/configs/agent_template.txt。
        English: With default config this reads
        forensic_agent/configs/agent_template.txt.
        """
        self.agent_config = agent_config

        self.agent_template_path = self._get_template_path(self.CONFIG_AGENT_TEMPLATE, self.DEFAULT_AGENT_TEMPLATE_FILE)

        self._agent_template = self._load_template(self.agent_template_path, "Agent模板")

        logger.info("ForensicTemplate initialized.")

    def _get_template_path(self, config_key: str, default_filename: str) -> Path:
        """Resolve a template path.

        中文说明: 用户传入路径按当前工作目录解析, 默认路径按包内 configs 解析。
        English: User paths resolve from the current working directory; default
        paths resolve inside the package configs directory.
        """
        custom_path = self.agent_config.get(config_key)
        if custom_path:
            return Path(custom_path)
        else:
            return Path(__file__).parent.parent / "configs" / default_filename

    def _load_template(self, template_path: Path, template_name: str) -> str:
        """Load template content.

        中文说明: 空模板会立即报错, 避免 Agent 启动后才进入无效循环。
        English: Empty templates fail fast so the agent does not start an
        invalid loop.
        """
        if template_path.exists():
            return self._read_template_file(template_path, template_name)
        else:
            logger.error(f"{template_name}文件不存在: {template_path}")
            raise ForensicTemplateError(f"{template_name}文件不存在: {template_path}")

    def _read_template_file(self, template_path: Path, template_name: str) -> str:
        """Read one UTF-8 template file.

        中文说明: 所有开源配置模板都要求 UTF-8 编码。
        English: All open-source config templates are expected to be UTF-8.
        """
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
        """Build a LangChain PromptTemplate.

        中文说明: tools/tool_names 由 LangGraph ReAct 节点自动注入, 不作为外部输入。
        English: tools/tool_names are injected by the LangGraph ReAct node and
        are not external inputs.
        """
        try:
            all_variables = self._extract_template_variables(self._agent_template)
            input_variables = [v for v in all_variables if v not in exclude_variables]
            logger.debug(f"提取的模板变量: {input_variables}")
            return PromptTemplate(template=self._agent_template, input_variables=input_variables)
        except Exception as e:
            logger.error(f"创建Agent模板失败: {e}")
            raise ForensicTemplateError(f"创建Agent模板失败: {e}") from e

    def _extract_template_variables(self, template_str: str) -> List[str]:
        """Extract format variables from a template string.

        中文说明: 只提取基础字段名, 支持 LangChain 模板变量检查。
        English: Extracts only base field names for LangChain template checks.
        """
        if not template_str:
            return []

        formatter = Formatter()
        variables = []

        try:
            for _, field_name, _, _ in formatter.parse(template_str):
                if field_name is not None and field_name:
                    base_name = field_name.split(".")[0].split("[")[0]
                    if base_name and base_name not in variables:
                        variables.append(base_name)
        except Exception as e:
            logger.warning(f"解析模板变量失败: {e}")
            return ["input", "tools", "current_stage", "messages", "agent_scratchpad"]

        return variables

    def format_system_message(self, goal: str) -> SystemMessage:
        """Return a simple fallback system message.

        中文说明: 兼容旧接口, 当前最小流程不主动调用。
        English: Kept for old interface compatibility; the minimal runtime does
        not call it directly.
        """
        try:
            formatted_content = self._agent_template.format(goal=goal.strip())
            return SystemMessage(content=formatted_content)
        except Exception as e:
            logger.error(f"格式化系统消息失败: {e}")
            return SystemMessage(content=f"你是一个专业的AIGC图像取证分析师。任务: {goal}")

    @property
    def agent_template(self) -> str:
        """Return the loaded agent template.

        中文说明: 供调试和测试读取。
        English: Exposed for debugging and tests.
        """
        return self._agent_template

    @property
    def system_template(self) -> str:
        """Return the loaded template through the legacy property.

        中文说明: 最小版没有独立 system template, 因此返回 agent template。
        English: The minimal release has no separate system template, so this
        returns the agent template.
        """
        return self._agent_template

    def reload_templates(self) -> None:
        """Reload template content from disk.

        中文说明: 修改提示词后可在测试中调用该方法重新加载。
        English: Tests may call this after editing prompt files.
        """
        self._agent_template = self._load_template(self.agent_template_path, "Agent模板")
        logger.info("Templates reloaded.")

    def get_template_info(self) -> Dict[str, Any]:
        """Return template diagnostics.

        中文说明: 不包含模板正文, 避免日志泄露用户自定义提示内容。
        English: Does not include template body so custom prompt content is not
        leaked in diagnostics.
        """
        return {
            "agent_template_path": str(self.agent_template_path),
            "agent_template_exists": self.agent_template_path.exists(),
            "agent_template_loaded": bool(self._agent_template),
            "agent_variables": self._extract_template_variables(self._agent_template),
        }

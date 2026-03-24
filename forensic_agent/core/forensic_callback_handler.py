import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import time
from dataclasses import dataclass, field
from loguru import logger

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration


@dataclass
class PerformanceStats:
    """性能统计数据结构"""

    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    llm_time: float = 0.0
    tool_time: float = 0.0


@dataclass
class ErrorStats:
    """错误统计数据结构"""

    llm_errors: int = 0
    tool_errors: int = 0
    error_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionContext:
    """执行上下文"""

    current_tool: Optional[str] = None
    current_action: Optional[str] = None
    step_start_time: Optional[float] = None


class ForensicCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        session_id: Optional[str] = None,
        max_content_length: int = 4096,
        enable_performance_tracking: bool = True,
        save_raw_data: bool = False,
    ):
        super().__init__()
        self.session_id = session_id or f"forensic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_content_length = max_content_length
        self.enable_performance_tracking = enable_performance_tracking
        self.save_raw_data = save_raw_data

        # 使用dataclass简化数据结构
        self.performance_stats = PerformanceStats()
        self.error_stats = ErrorStats()
        self.current_context = ExecutionContext()

        logger.info(f"ForensicCallbackHandler初始化 - Session: {self.session_id}")

    def _truncate_content(self, content: str) -> str:
        """截取内容避免过长"""
        if not isinstance(content, str):
            content = str(content)
        return content[: self.max_content_length] + "..." if len(content) > self.max_content_length else content

    def _start_timing(self) -> None:
        """开始计时"""
        if self.enable_performance_tracking:
            self.current_context.step_start_time = time.time()
            if self.performance_stats.start_time is None:
                self.performance_stats.start_time = time.time()

    def _end_timing(self, operation_type: str) -> float:
        """结束计时并更新统计"""
        if not self.enable_performance_tracking or not self.current_context.step_start_time:
            return 0.0

        duration = time.time() - self.current_context.step_start_time

        if operation_type == "llm":
            self.performance_stats.llm_time += duration
        elif operation_type == "tool":
            self.performance_stats.tool_time += duration

        self.current_context.step_start_time = None
        return duration

    def _handle_error(self, error: Union[Exception, KeyboardInterrupt], error_type: str, context: str = "", *args, **kwargs) -> None:
        """统一错误处理"""
        if error_type == "llm":
            self.error_stats.llm_errors += 1
        elif error_type == "tool":
            self.error_stats.tool_errors += 1

        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self.error_stats.error_details.append(error_info)

        logger.error(f"[{error_type.upper()}] {context}: {str(error)}")

    # LLM 相关方法
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """处理LLM开始"""
        self._start_timing()
        self.performance_stats.llm_calls += 1

        model_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        logger.debug(f"LLM推理开始 - 模型: {model_name}, 提示数量: {len(prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """处理LLM结束"""
        duration = self._end_timing("llm")

        # 提取生成内容和token统计
        output_content = ""
        token_count = 0

        if response.generations and response.generations[0]:
            chat_generation: ChatGeneration = response.generations[0][0]
            output_content = chat_generation.message.content

            # 方法1: 从response.llm_output获取token统计（最准确）
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    token_count = token_usage.get("total_tokens", 0)
                    if token_count == 0:
                        # 如果没有total_tokens，尝试计算
                        prompt_tokens = token_usage.get("prompt_tokens", 0)
                        completion_tokens = token_usage.get("completion_tokens", 0)
                        token_count = prompt_tokens + completion_tokens

            # 方法2: 从generation的response_metadata获取
            if token_count == 0 and hasattr(chat_generation.message, "response_metadata"):
                response_metadata = chat_generation.message.response_metadata
                if "token_usage" in response_metadata:
                    usage = response_metadata["token_usage"]
                    token_count = usage.get("total_tokens", 0)

            if token_count == 0 and hasattr(chat_generation.message, "usage_metadata"):
                usage_metadata = chat_generation.message.usage_metadata or {}
                if "total_tokens" in usage_metadata:
                    token_count = usage_metadata.get("total_tokens", 0)

            # 方法4: 简单估算（不准确，仅作备用）
            if token_count == 0 and output_content:
                # 粗略估算：英文约4字符=1token，中文约1.5字符=1token
                char_count = len(output_content)
                # 简单估算，可根据实际模型调整
                token_count = max(1, char_count // 3)

        # 更新token统计
        if token_count > 0:
            self.performance_stats.total_tokens += token_count
            logger.debug(f"统计到tokens: {token_count}")

        logger.debug(f"LLM推理完成 - 用时: {duration:.2f}s, 内容: {self._truncate_content(output_content)}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """处理LLM错误"""
        self._end_timing("llm")
        self._handle_error(error, "llm", "LLM执行过程中")

    # Agent 相关方法
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """处理Agent动作"""
        self.current_context.current_action = action.tool
        logger.info(f"执行动作: {action.tool} - 输入: {self._truncate_content(str(action.tool_input))}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """处理Agent完成"""
        self.performance_stats.end_time = time.time()
        output = finish.return_values.get("output", "")
        logger.info(f"Agent完成 - 输出: {self._truncate_content(output)}")

    # Tool 相关方法
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """处理工具开始"""
        self._start_timing()
        self.performance_stats.tool_calls += 1

        tool_name = serialized.get("name", "未知工具")
        self.current_context.current_tool = tool_name
        logger.debug(f"开始调用工具: {tool_name}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """处理工具结束"""
        duration = self._end_timing("tool")
        tool_name = self.current_context.current_tool or "未知工具"
        logger.debug(f"工具完成: {tool_name} - 用时: {duration:.2f}s")
        self.current_context.current_tool = None

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """处理工具错误"""
        self._end_timing("tool")
        tool_name = self.current_context.current_tool or "未知工具"
        self._handle_error(error, "tool", f"工具 {tool_name} 执行过程中")
        self.current_context.current_tool = None

    # Chain 相关方法
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """处理链开始"""
        if serialized:
            chain_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        else:
            chain_name = kwargs["name"]
        logger.debug(f"链开始: {chain_name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """处理链结束"""
        logger.debug("链完成")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """处理链错误"""
        self._handle_error(error, "chain", "链执行过程中")

        # 打印详细的错误堆栈
        logger.error("=" * 80)
        logger.error("链执行错误详细堆栈:")
        logger.error("-" * 80)

        if isinstance(error, KeyboardInterrupt):
            logger.error("用户中断执行")
        else:
            # 额外记录关键信息
            logger.exception("异常堆栈信息:")
            # 如果有kwargs中的额外上下文信息也记录下来
            if kwargs:
                logger.error(f"上下文信息: {kwargs}")

        logger.error("=" * 80)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """处理文本输出"""
        if text.strip():
            logger.debug(f"推理: {self._truncate_content(text.strip())}")

    # 数据管理方法
    def clear(self) -> None:
        """清空所有记录"""
        self.performance_stats = PerformanceStats()
        self.error_stats = ErrorStats()
        self.current_context = ExecutionContext()
        logger.debug("记录已清空")

    def get_summary(self) -> Dict[str, Any]:
        """获取完整摘要"""
        total_time = 0
        if self.performance_stats.start_time and self.performance_stats.end_time:
            total_time = self.performance_stats.end_time - self.performance_stats.start_time

        return {
            "session_id": self.session_id,
            "performance": {
                "total_time": total_time,
                "llm_calls": self.performance_stats.llm_calls,
                "tool_calls": self.performance_stats.tool_calls,
                "total_tokens": self.performance_stats.total_tokens,
                "llm_time": self.performance_stats.llm_time,
                "tool_time": self.performance_stats.tool_time,
            },
            "errors": {
                "llm_errors": self.error_stats.llm_errors,
                "tool_errors": self.error_stats.tool_errors,
                "total_errors": self.error_stats.llm_errors + self.error_stats.tool_errors,
            },
        }

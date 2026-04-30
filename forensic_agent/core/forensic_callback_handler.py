"""LangChain callback utilities for AgentFoX runs.

中文说明: 该模块只记录推理耗时、工具调用和错误摘要, 不保存图片、密钥或原始私有数据。
English: This module records timing, tool calls, and error summaries only; it
does not persist images, credentials, or private raw data.
"""

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
    """Runtime performance counters.

    中文说明: 统计信息只用于结果 JSON 中的轻量 metrics 字段。
    English: These counters are used only for the lightweight metrics field in
    result JSON.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    llm_time: float = 0.0
    tool_time: float = 0.0


@dataclass
class ErrorStats:
    """Runtime error counters.

    中文说明: 错误明细不包含图片内容, 只记录异常类型和文本。
    English: Error details do not contain image content; only exception type
    and text are recorded.
    """

    llm_errors: int = 0
    tool_errors: int = 0
    error_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionContext:
    """Current callback execution context.

    中文说明: 用于把耗时归因到当前 LLM 或工具调用。
    English: Used to attribute timing to the active LLM or tool call.
    """

    current_tool: Optional[str] = None
    current_action: Optional[str] = None
    step_start_time: Optional[float] = None


class ForensicCallbackHandler(BaseCallbackHandler):
    """Collect lightweight callback telemetry.

    中文说明: model_name 是可选字段, 兼容 ForensicAgent 为每个 workflow 传入模型名。
    English: model_name is optional and supports per-workflow metadata passed by
    ForensicAgent.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        max_content_length: int = 4096,
        enable_performance_tracking: bool = True,
        save_raw_data: bool = False,
    ):
        super().__init__()
        self.session_id = session_id or f"forensic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_name = model_name
        self.max_content_length = max_content_length
        self.enable_performance_tracking = enable_performance_tracking
        self.save_raw_data = save_raw_data

        self.performance_stats = PerformanceStats()
        self.error_stats = ErrorStats()
        self.current_context = ExecutionContext()

        model_suffix = f", Model: {self.model_name}" if self.model_name else ""
        logger.info(f"ForensicCallbackHandler initialized - Session: {self.session_id}{model_suffix}")

    def _truncate_content(self, content: str) -> str:
        """Truncate long log content.

        中文说明: 避免日志输出完整长回复或大对象。
        English: Prevents logs from emitting full long responses or large objects.
        """
        if not isinstance(content, str):
            content = str(content)
        return content[: self.max_content_length] + "..." if len(content) > self.max_content_length else content

    def _start_timing(self) -> None:
        """Start timing the current callback span.

        中文说明: 第一次调用同时记录整轮推理开始时间。
        English: The first call also marks the start of the whole inference run.
        """
        if self.enable_performance_tracking:
            self.current_context.step_start_time = time.time()
            if self.performance_stats.start_time is None:
                self.performance_stats.start_time = time.time()

    def _end_timing(self, operation_type: str) -> float:
        """Finish timing and update counters.

        中文说明: operation_type 只能归入 llm 或 tool 两类耗时。
        English: operation_type is attributed to either llm or tool timing.
        """
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
        """Record one runtime error.

        中文说明: 只记录异常摘要, 不把图片或 API key 写入日志结构。
        English: Only an exception summary is recorded; images and API keys are
        never written into the log structure.
        """
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

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Handle LLM start callbacks.

        中文说明: 只记录调用次数和提示数量。
        English: Records only call count and prompt count.
        """
        self._start_timing()
        self.performance_stats.llm_calls += 1

        model_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        logger.debug(f"LLM推理开始 - 模型: {model_name}, 提示数量: {len(prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Handle LLM end callbacks.

        中文说明: 优先读取模型返回的 token 统计, 不存在时才做粗略估算。
        English: Prefers model-provided token usage and falls back to a rough
        estimate only when needed.
        """
        duration = self._end_timing("llm")

        output_content = ""
        token_count = 0

        if response.generations and response.generations[0]:
            chat_generation: ChatGeneration = response.generations[0][0]
            output_content = chat_generation.message.content

            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    token_count = token_usage.get("total_tokens", 0)
                    if token_count == 0:
                        prompt_tokens = token_usage.get("prompt_tokens", 0)
                        completion_tokens = token_usage.get("completion_tokens", 0)
                        token_count = prompt_tokens + completion_tokens

            if token_count == 0 and hasattr(chat_generation.message, "response_metadata"):
                response_metadata = chat_generation.message.response_metadata
                if "token_usage" in response_metadata:
                    usage = response_metadata["token_usage"]
                    token_count = usage.get("total_tokens", 0)

            if token_count == 0 and hasattr(chat_generation.message, "usage_metadata"):
                usage_metadata = chat_generation.message.usage_metadata or {}
                if "total_tokens" in usage_metadata:
                    token_count = usage_metadata.get("total_tokens", 0)

            if token_count == 0 and output_content:
                char_count = len(output_content)
                token_count = max(1, char_count // 3)

        if token_count > 0:
            self.performance_stats.total_tokens += token_count
            logger.debug(f"统计到tokens: {token_count}")

        logger.debug(f"LLM推理完成 - 用时: {duration:.2f}s, 内容: {self._truncate_content(output_content)}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Handle LLM errors.

        中文说明: LLM 异常会计入本轮 metrics, 并继续交给上层异常链处理。
        English: LLM errors are counted in metrics and still propagate through
        the caller's error path.
        """
        self._end_timing("llm")
        self._handle_error(error, "llm", "LLM执行过程中")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Handle an agent action.

        中文说明: 记录工具名和截断后的输入, 不记录图片二进制内容。
        English: Logs the tool name and truncated input, not binary image data.
        """
        self.current_context.current_action = action.tool
        logger.info(f"执行动作: {action.tool} - 输入: {self._truncate_content(str(action.tool_input))}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Handle agent finish.

        中文说明: 标记整轮推理结束时间。
        English: Marks the end time of the whole inference run.
        """
        self.performance_stats.end_time = time.time()
        output = finish.return_values.get("output", "")
        logger.info(f"Agent完成 - 输出: {self._truncate_content(output)}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Handle tool start callbacks.

        中文说明: 工具调用计数用于判断 Agent 是否实际使用了语义分析工具。
        English: Tool call count helps confirm whether the semantic analysis
        tool was actually used.
        """
        self._start_timing()
        self.performance_stats.tool_calls += 1

        tool_name = serialized.get("name", "未知工具")
        self.current_context.current_tool = tool_name
        logger.debug(f"开始调用工具: {tool_name}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Handle tool end callbacks.

        中文说明: 输出内容不在 info 日志完整打印, 避免报告过长。
        English: Tool output is not fully printed at info level to avoid long logs.
        """
        duration = self._end_timing("tool")
        tool_name = self.current_context.current_tool or "未知工具"
        logger.debug(f"工具完成: {tool_name} - 用时: {duration:.2f}s")
        self.current_context.current_tool = None

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Handle tool errors.

        中文说明: 记录当前工具名后清理上下文, 防止影响后续工具统计。
        English: Records the active tool name and clears context so later tool
        metrics are not polluted.
        """
        self._end_timing("tool")
        tool_name = self.current_context.current_tool or "未知工具"
        self._handle_error(error, "tool", f"工具 {tool_name} 执行过程中")
        self.current_context.current_tool = None

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Handle chain start callbacks.

        中文说明: 某些 LangGraph 回调没有 serialized, 因此安全读取 name。
        English: Some LangGraph callbacks omit serialized, so name is read safely.
        """
        if serialized:
            chain_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        else:
            chain_name = kwargs.get("name", "unknown")
        logger.debug(f"链开始: {chain_name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Handle chain end callbacks.

        中文说明: 当前只做 debug 记录。
        English: Currently used for debug logging only.
        """
        logger.debug("链完成")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Handle chain errors with stack details.

        中文说明: 这里记录堆栈方便定位配置/工具错误, 但仍不写入敏感配置值。
        English: Stack details help diagnose config/tool errors while avoiding
        sensitive config values.
        """
        self._handle_error(error, "chain", "链执行过程中")

        logger.error("=" * 80)
        logger.error("链执行错误详细堆栈:")
        logger.error("-" * 80)

        if isinstance(error, KeyboardInterrupt):
            logger.error("用户中断执行")
        else:
            logger.exception("异常堆栈信息:")
            if kwargs:
                logger.error(f"上下文信息: {kwargs}")

        logger.error("=" * 80)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Handle streaming text callbacks.

        中文说明: 流式输出只进入 debug 日志并进行长度截断。
        English: Streaming output goes only to debug logs and is truncated.
        """
        if text.strip():
            logger.debug(f"推理: {self._truncate_content(text.strip())}")

    def clear(self) -> None:
        """Clear all callback counters.

        中文说明: 每张图片推理前调用, 保证 metrics 不串样本。
        English: Called before each image inference so metrics do not leak
        across samples.
        """
        self.performance_stats = PerformanceStats()
        self.error_stats = ErrorStats()
        self.current_context = ExecutionContext()
        logger.debug("记录已清空")

    def get_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable telemetry summary.

        中文说明: 该结果会随 final_output 保存, 便于用户确认运行耗时和错误数。
        English: The result is saved with final_output so users can inspect
        runtime cost and error counts.
        """
        total_time = 0
        if self.performance_stats.start_time and self.performance_stats.end_time:
            total_time = self.performance_stats.end_time - self.performance_stats.start_time

        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
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

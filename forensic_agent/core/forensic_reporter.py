"""Final structured report generation.

中文说明: Reporter 将 Agent 对话历史压缩为最终 JSON 结论, 并兼容没有 expert tool 的最小配置。
English: The reporter converts agent conversation history into a final JSON
verdict and supports minimal configs without expert tools.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from .agent_state import CustomAgentState
from .forensic_dataclass import FinalResponse


class ForensicReporter:
    """Generate and validate the final forensic verdict.

    中文说明: 该类只依赖已有对话和工具输出, 不再调用已删除的 profile/专家代码。
    English: This class depends only on existing conversation and tool outputs;
    it no longer calls removed profile or expert code.
    """

    USER_PROMPT_TEMPLATE = """
You are a digital image forensics arbitrator. Analyze the provided conversation history and produce a structured forensic report.

Decision categories:
- 0 = Authentic camera-captured image
- 1 = AI-generated or forged image

Conversation history:
{conversation_history}

Requirements:
- Trace every claim to the conversation history.
- Treat semantic analysis as the main evidence in the minimal open-source runtime.
- If evidence is insufficient, explain the uncertainty and still provide the best supported binary decision.
"""

    FORMAT_PROMPT_TEMPLATE = """
Please review the consistency of the forensic report and provide the final forensic finding.

Previous content:
{previous_report_content}
"""

    def __init__(self, llm_config: dict | None = None, llm: ChatOpenAI | None = None, self_expressing: bool = False):
        config = dict(llm_config or {})
        prompt_path = Path(config.pop("prompt_path"))
        if not prompt_path.exists():
            raise FileNotFoundError(f"Reporter prompt file not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        self.llm = llm if not config else init_chat_model(**config)
        if self.llm is None:
            raise ValueError("A reporter LLM instance is required.")
        self.self_expressing = self_expressing
        self.llm_with_structured = self.llm.with_structured_output(FinalResponse)
        logger.info("ForensicReporter initialized.")

    def generate_report(self, state: CustomAgentState) -> CustomAgentState:
        """Generate final_response and update state.

        中文说明: expert_result 在最小配置中可能不存在, 此时写入空字典而不是报错。
        English: expert_result may be absent in minimal config; in that case an
        empty dict is written instead of raising an error.
        """
        if self.self_expressing is False:
            conversation_history = self._extract_conversation_history(state["messages"][2:-1])
            report_prompt = self.USER_PROMPT_TEMPLATE.format(conversation_history=conversation_history)
            raw_report = self.llm.invoke([HumanMessage(content=report_prompt)]).content
        else:
            conversation_history = self._extract_conversation_history(state["messages"], target_role=["AIMessage"])
            raw_report = self.FORMAT_PROMPT_TEMPLATE.format(previous_report_content=conversation_history)

        final_response = self.llm_with_structured.invoke([HumanMessage(content=raw_report)]).model_dump()
        final_response["semantic_result"] = self._load_tool_result(state, "semantic_analysis")
        final_response["expert_result"] = self._load_tool_result(state, "expert_results")
        final_response["detail_report"] = conversation_history

        state["final_response"] = final_response
        if final_response.get("is_success"):
            state["messages"].append(AIMessage(content="update stage to: done"))
        else:
            state["messages"].extend(
                [
                    ToolMessage(
                        content="No clear conclusion has been reached. Please provide a final report.",
                        name="report_conflict_tool",
                        tool_call_id=str(uuid.uuid4()),
                    ),
                    AIMessage(content="update stage to: finally_report"),
                ]
            )
        state.pop("remaining_steps", None)
        return state

    def _load_tool_result(self, state: CustomAgentState, tool_name: str) -> dict[str, Any]:
        """Load the latest JSON-like result from a tool message.

        中文说明: LangChain 可能把工具返回值保存成 dict 或 JSON 字符串, 两种都兼容。
        English: LangChain may store tool outputs as dicts or JSON strings; both
        forms are supported.
        """
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == tool_name:
                content = msg.content
                if isinstance(content, dict):
                    return content
                if not content:
                    return {}
                try:
                    parsed = json.loads(content)
                    return parsed if isinstance(parsed, dict) else {"value": parsed}
                except json.JSONDecodeError:
                    return {"text": str(content)}
        return {}

    def _extract_conversation_history(self, messages, target_role=None) -> str:
        """Collect non-empty non-human messages for the reporter.

        中文说明: 去重后生成紧凑历史, 减少最终报告阶段 token 压力。
        English: A compact deduplicated history reduces token pressure in the
        final reporting step.
        """
        allowed_roles = set(target_role or [])
        history_parts = []
        seen_contents = set()
        skip_next = False

        for msg in messages:
            if skip_next:
                skip_next = False
                continue
            role = type(msg).__name__
            content = getattr(msg, "content", None)
            reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content")

            if isinstance(content, str) and content.startswith("Unrecognized stage: "):
                skip_next = True
                continue
            if role == "HumanMessage":
                continue
            if isinstance(content, str) and content.lower().strip() == "update stage to: done":
                continue
            if allowed_roles and role not in allowed_roles:
                continue
            if not content and not reasoning:
                continue

            content_str = json.dumps(content, ensure_ascii=False, sort_keys=True) if isinstance(content, (dict, list)) else str(content)
            if reasoning:
                content_str += f"\n[Reasoning]: {reasoning}"
            unique_key = f"[{role}]:{content_str}"
            if unique_key not in seen_contents:
                seen_contents.add(unique_key)
                history_parts.append(unique_key)

        if not history_parts:
            raise ValueError("Conversation history is insufficient for report generation.")
        return "\n\n".join(history_parts)


def create_reporter_node(config: dict, llm: ChatOpenAI, self_expressing: bool = False) -> Any:
    """Create a StateGraph reporter node.

    中文说明: ForensicAgent 将该函数返回值注册为 finally_report 节点。
    English: ForensicAgent registers the returned callable as the finally_report
    node.
    """
    reporter = ForensicReporter(config, llm, self_expressing)

    def reporter_node(state: CustomAgentState) -> Dict[str, Any]:
        return reporter.generate_report(state)

    return reporter_node

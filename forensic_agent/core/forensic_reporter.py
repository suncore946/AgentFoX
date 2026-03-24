from pathlib import Path
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger
import uuid
import json
from langchain.chat_models import init_chat_model
from .forensic_dataclass import FinalResponse
from .agent_state import CustomAgentState


class ForensicReporter:

    USER_PROMPT_TEMPLATE = """
You are a digital image forensics arbitrator. Analyze the provided conversation history and produce a structured forensic report to determine image authenticity.

## DECISION CATEGORIES:
- 0 = Authentic (camera-captured)
- 1 = Forged (AI-generated)

## FORENSIC ANALYSIS FRAMEWORK:

### Evidence Source
```
{conversation_history}
```

### 1. KEY OBSERVATIONS
Extract and document critical findings:
- **Semantic-Level Analysis**: Identifies physical law violations and semantic inconsistencies. Note: Advanced AI generation may circumvent traditional flaws, requiring multi-method verification.
- **Pixel-Level Predictions**: Pred_prob values closer to 1.0 suggest AI generation; values closer to 0.0 suggest authenticity. These represent model judgments based on training data, not confidence levels—integrate with other evidence.
- **Clustering Profile Intelligence**: Provides supplementary insights through feature analysis. Performance variations between models may be subtle—use as supporting evidence only.

### 2. EVIDENCE SYNTHESIS
- **Multi-dimensional Assessment**: Integrate findings across all analysis methods
- **Reliability Weighting**: Assess credibility of each evidence source
- **Comprehensive Reasoning**: Draw conclusions from holistic evidence review

### 3: confidence_assessment
Objective: Conduct comprehensive analysis of all preceding content to evaluate confidence levels.
- Instead of invoking tools, make inferences based on the existing information.
- aggregate results, perform consistency analysis, and present confidence assessment outcomes for each dimension along with supporting justifications.

### 4. FORENSIC DETERMINATION
**Result summary: The semantics and forensic results of each expert model**
**Confidence estimation: Possible confidence level estimation for each dimension(semantic, expert, cluster strength)**
**Justification: Detailed reasoning process for the final decision (Majority rule or minority rule or other)**
**Final Decision: [0 / 1]**

Confirm before finalizing:
- ☐ Accurate representation of all model predictions
- ☐ Proper interpretation of prediction confidence levels
- ☐ Sufficient forensic justification for consensus deviations
- ☐ Direct traceability of claims to conversation history
- ☐ Adequate treatment of contradictory evidence
- ☐ When you decide that the minority should prevail over the majority, is there sufficient justification for such a decision?
- ☐ Is the conclusion consistent with your analysis and reasoning?
"""

    FORMAT_PROMPT_TEMPLATE = """
Please review the consistency of the forensic report and provide the final forensic findings.

Previous Content:
{previous_report_content}
"""

    def __init__(self, llm_config: dict = None, llm: ChatOpenAI = None, self_expressing: bool = False):
        """
        初始化取证报告生成器

        Args:
            config: 配置参数字典，包含报告生成的相关配置
            llm: 用于生成报告的语言模型
        """
        target_config = llm_config.copy()
        prompt_path = Path(target_config.pop("prompt_path"))
        assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        if not target_config:
            self.llm = llm
        else:
            self.llm = init_chat_model(**target_config)

        self.self_expressing = self_expressing
        self.llm_with_structured = self.llm.with_structured_output(FinalResponse)
        logger.info("ForensicReporter 初始化完成")

    def generate_report(self, state: CustomAgentState) -> CustomAgentState:
        """
        生成最终取证报告

        Args:
            state: Agent的状态对象，包含所有历史消息和分析结果

        Returns:
            更新后的状态字典
        """

        if self.self_expressing is False:
            conversation_history = self._extract_conversation_history(state["messages"][2:-1])
            # 构建报告生成提示
            report_prompt = self.USER_PROMPT_TEMPLATE.format(conversation_history=conversation_history)
            # 调用LLM生成结构化报告
            # logger.info(f"开始生成最终取证报告: {state['image_path']}")
            res = self.llm.invoke([HumanMessage(content=report_prompt)], **self.self_expressing)
            # 要求输出序列化的FinalResponse对象
            detection_report = res.content
        else:
            # 自我表达式下，直接使用对话历史作为输入
            conversation_history = self._extract_conversation_history(state["messages"], target_role=["AIMessage"])
            detection_report = self.FORMAT_PROMPT_TEMPLATE.format(previous_report_content=conversation_history)

        # max_retry = 3
        # raw = None
        # msg = [HumanMessage(content=f"{detection_report}\n\n{self.format_instructions}")]
        # for attempt in range(1, max_retry + 1):
        #     try:
        #         raw = self.llm.invoke(msg)
        #         final_response = self.parser.parse(raw.content).model_dump()
        #         break
        #     except Exception as err:
        #         logger.warning(f"parse failed ({attempt}/{max_retry}): {err}")
        #         detection_report = f"Please fix: {err}"
        #         if raw and getattr(raw, "content", None):
        #             msg.append(AIMessage(content=raw.content))
        #             msg.append(HumanMessage(content=detection_report))
        # else:
        #     raise RuntimeError("FinalResponse parsing failed")

        final_response = self.llm_with_structured.invoke([HumanMessage(content=detection_report)]).model_dump()
        final_response["semantic_result"] = json.loads(self.get_special_tool_results(state, "semantic_analysis"))
        final_response["expert_result"] = json.loads(self.get_special_tool_results(state, "expert_results"))
        final_response["detail_report"] = conversation_history

        state["final_response"] = final_response
        if final_response["is_success"]:
            state["messages"].append(AIMessage(content="update stage to: done"))
        else:
            state["messages"].extend(
                [
                    ToolMessage(
                        content="No clear conclusion has been reached. Please provide a finally report.",
                        name="report_conflict_tool",
                        tool_call_id=str(uuid.uuid4()),
                    ),
                    AIMessage(content="update stage to: finally_report"),
                ]
            )

        state.pop("remaining_steps", None)
        return state

    def get_special_tool_results(self, state: CustomAgentState, tool_name: str) -> Any:
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == tool_name:
                return msg.content
        return None

    def _extract_conversation_history(self, messages, target_role=None) -> str:
        if target_role is None:
            target_role = []

        allowed_roles = set(target_role)
        history_parts = []
        seen_contents = set()
        skip_next = False

        for msg in messages:
            if skip_next:
                skip_next = False
                continue

            role = type(msg).__name__
            content = getattr(msg, "content", None)
            thinks = getattr(msg, "additional_kwargs", {}).get("reasoning_content")

            # 跳过未识别阶段的消息
            if isinstance(content, str) and content.startswith("Unrecognized stage: "):
                skip_next = True
                continue
            # 跳过空内容消息
            if not content and not thinks:
                continue
            # 跳过特定角色或内容
            if role == "HumanMessage":
                continue
            # 跳过仅有更新阶段的消息
            if isinstance(content, str) and content.lower() == "update stage to: done":
                continue
            # 过滤角色
            if allowed_roles and role not in allowed_roles:
                continue

            # 构建唯一内容标识，避免重复
            content_str = json.dumps(content, ensure_ascii=False, sort_keys=True) if isinstance(content, (dict, list)) else str(content)
            if thinks:
                thinks_str = json.dumps(thinks, ensure_ascii=False, sort_keys=True) if isinstance(thinks, (dict, list)) else str(thinks)
                content_str += f"[Reasoning]: {thinks_str}"

            unique_key = f"[{role}]:{content_str}"
            if unique_key not in seen_contents:
                seen_contents.add(unique_key)
                history_parts.append(unique_key)

        if not history_parts:
            raise ValueError("Conversation history is insufficient for report generation.")

        return "\n\n".join(history_parts)


# 用于集成到 ForensicAgent 的节点函数
def create_reporter_node(config: dict, llm: ChatOpenAI, self_expressing=False) -> Any:
    """
    创建reporter节点的工厂函数

    Args:
        config: 配置字典
        llm: 语言模型实例

    Returns:
        可用于StateGraph的节点函数
    """
    reporter = ForensicReporter(config, llm, self_expressing)

    def reporter_node(state: CustomAgentState) -> Dict[str, Any]:
        # 生成最终报告
        state: CustomAgentState = reporter.generate_report(state)
        return state

    return reporter_node

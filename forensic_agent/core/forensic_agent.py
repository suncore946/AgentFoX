"""LangGraph ReAct agent for AgentFoX.

中文说明: 该模块负责把 LLM、工具和最终 reporter 串成可执行状态图。
English: This module connects the LLM, tools, and final reporter into an
executable state graph.
"""

from __future__ import annotations

import atexit
import json
import re
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

from ..manager.image_manager import ImageManager
from .agent_state import CustomAgentState, StageEnum
from .forensic_callback_handler import ForensicCallbackHandler
from .forensic_llm import ForensicLLM
from .forensic_reporter import create_reporter_node
from .forensic_template import ForensicTemplate
from .forensic_tools import ForensicTools


class WorkflowExecutor:
    """Small per-workflow thread executor.

    中文说明: 最小版默认串行运行, 但保留该执行器方便未来多 LLM 扩展。
    English: The minimal release runs serially by default, but this executor is
    kept for future multi-LLM expansion.
    """

    def __init__(self, worker_count: Dict[int, int], worker_fn: Callable[[int, Tuple[Any, ...]], None]):
        self._queues = {workflow_id: Queue() for workflow_id in worker_count}
        self._threads = {workflow_id: [] for workflow_id in worker_count}
        self._shutdown = False
        self._running = False
        self._worker_fn = worker_fn
        self._worker_count = worker_count

    def start(self) -> None:
        """Start worker threads.

        中文说明: 已启动时重复调用会被忽略。
        English: Repeated calls are ignored when workers are already running.
        """
        if self._running:
            return
        self._shutdown = False
        for workflow_id, count in self._worker_count.items():
            for index in range(count):
                thread = Thread(target=self._run, name=f"Workflow-{workflow_id}-Worker-{index}", args=(workflow_id,), daemon=True)
                thread.start()
                self._threads[workflow_id].append(thread)
        self._running = True

    def shutdown(self, wait: bool = True) -> None:
        """Stop worker threads.

        中文说明: 进程退出时也会调用, 避免后台线程悬挂。
        English: This is also called at process exit to avoid dangling threads.
        """
        if not self._running:
            return
        self._shutdown = True
        for workflow_id, queue in self._queues.items():
            for _ in self._threads[workflow_id]:
                queue.put(None)
        if wait:
            for threads in self._threads.values():
                for thread in threads:
                    thread.join(timeout=5.0)
        for threads in self._threads.values():
            threads.clear()
        self._running = False

    def submit(self, workflow_id: int, *payload: Any) -> None:
        """Submit one task payload.

        中文说明: workflow_id 必须对应已配置的 LLM。
        English: workflow_id must match a configured LLM.
        """
        if not self._running or self._shutdown:
            raise RuntimeError("Workflow executor is not running.")
        queue = self._queues.get(workflow_id)
        if queue is None:
            raise KeyError(f"Unknown workflow id: {workflow_id}")
        queue.put(payload)

    @property
    def is_running(self) -> bool:
        """Return executor status.

        中文说明: shutdown 后即使线程列表未清空也视为不可用。
        English: After shutdown, the executor is considered unavailable even if
        thread cleanup is still in progress.
        """
        return self._running and not self._shutdown

    def _run(self, workflow_id: int) -> None:
        queue = self._queues[workflow_id]
        while True:
            try:
                task = queue.get(timeout=0.5)
            except Empty:
                if self._shutdown:
                    break
                continue
            try:
                if task is None:
                    break
                self._worker_fn(workflow_id, task)
            finally:
                queue.task_done()


class ForensicAgent:
    """Run the AgentFoX state machine for one image.

    中文说明: Agent 每轮必须输出 `update stage to: ...`, stage_check 据此推进流程。
    English: The agent must output `update stage to: ...`; stage_check uses that
    marker to advance the workflow.
    """

    QUESTION_PROMPT = """
Analyze the image to determine whether it is authentic or AI-generated. Use available tools, summarize evidence, and produce a final report.
"""

    CHECK_STAGE_ERROR_MSG = """Unrecognized stage. Reply with `update stage to: [next_state]`, `update stage to: done`, or `update stage to: error`. """

    def __init__(
        self,
        config: dict,
        forensic_tool: ForensicTools,
        forensic_llm: ForensicLLM,
        image_manager: ImageManager,
        is_debug: bool = False,
    ):
        self.agent_config = config or {}
        self.forensic_tool = forensic_tool
        self.forensic_llm = forensic_llm
        self.llms: List[BaseChatModel] = forensic_llm.llms
        self.image_manager = image_manager
        self.verbose = is_debug
        self.max_iterations = int(self.agent_config.get("max_iterations", 40))
        self.callback_handler = ForensicCallbackHandler()

        template = ForensicTemplate(self.agent_config)
        self.prompt_template: PromptTemplate = template.build_agent_template()

        self.workflow_info: Dict[int, StateGraph] = {}
        self.workflow_tools: Dict[int, List[Any]] = {}
        self.num_workflows = len(self.llms)
        for workflow_id, llm in enumerate(self.llms):
            tools = self.forensic_tool.get_tools_for_llm(llm)
            self.workflow_tools[workflow_id] = tools
            self.workflow_info[workflow_id] = self._build_graph(llm, tools)
        if self.num_workflows == 0:
            raise ValueError("No LLM workflow was constructed.")

        self.per_workflow_workers = int(self.agent_config.get("per_workflow_workers", 1))
        self._executor = WorkflowExecutor(
            {workflow_id: self.per_workflow_workers for workflow_id in range(self.num_workflows)},
            self._dispatch_task,
        )
        self._next_workflow = 0
        self._lock = Lock()
        atexit.register(lambda: self.shutdown_workers(wait=False))
        logger.info("AgentFoX forensic agent initialized.")

    def _dispatch_task(self, workflow_id: int, payload: Tuple[Any, ...]) -> None:
        image_path, result_queue, error_queue = payload
        try:
            result_queue.put((image_path, self.think_and_act(image_path, workflow_id)))
        except Exception as exc:
            error_queue.put((image_path, exc))

    def _get_workflow_tools(self, workflow_id: Optional[int] = None) -> List[Any]:
        """Return tools for a workflow.

        中文说明: 多 LLM 时每个 workflow 持有独立工具实例。
        English: In multi-LLM mode each workflow owns independent tool instances.
        """
        if workflow_id is not None and workflow_id in self.workflow_tools:
            return self.workflow_tools[workflow_id]
        return self.forensic_tool.get_all_tools()

    def stage_check(self, state: CustomAgentState) -> Dict[str, Any]:
        """Parse stage transition from the latest AI message.

        中文说明: 解析失败时会把可用 stage/tool 反馈给 Agent 让其自修正。
        English: If parsing fails, available stages/tools are fed back so the
        agent can self-correct.
        """

        def _check_stage(messages: List[AnyMessage]) -> StageEnum:
            available_stages = ", ".join(stage.value for stage in StageEnum)
            available_tools = ", ".join(tool.name for tool in self._get_workflow_tools(state.get("workflow_id")))
            error_msg = f"{self.CHECK_STAGE_ERROR_MSG} Available stages: {available_stages}. Available tools: {available_tools}."
            if not messages:
                raise ValueError(error_msg)

            ai_msg = next(
                (message for message in reversed(messages) if isinstance(message, AIMessage) and getattr(message, "content", None)),
                None,
            )
            if ai_msg is None:
                raise ValueError(error_msg)

            content = ai_msg.content
            text = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
            reasoning = ai_msg.additional_kwargs.get("reasoning_content", "")
            low = f"{reasoning}\n{text}".lower()
            matches = re.findall(r"update stage to:\s*'?([a-zA-Z0-9_]+)'?", low)
            if not matches:
                raise ValueError(error_msg)
            stage_key = matches[-1].upper()
            if stage_key in StageEnum.__members__:
                return StageEnum[stage_key]
            raise ValueError(error_msg)

        try:
            state["current_stage"] = _check_stage(state["messages"])
            if state["current_stage"] == StageEnum.FINALLY_REPORT:
                state["messages"].append(
                    HumanMessage(content='Do not use tools in this step. Produce the final report and end with "update stage to: done".')
                )
        except Exception as exc:
            logger.debug(f"Stage parsing failed: {exc}")
            state["messages"].append(HumanMessage(content=str(exc)))
        finally:
            state.pop("remaining_steps", None)
            iterations = len(state["messages"]) // 2
            if iterations >= self.max_iterations:
                state["current_stage"] = StageEnum.MAX_ITERATIONS
                state["messages"].append(AIMessage(content="Maximum iterations reached. update stage to: error"))
        return state

    def _build_graph(self, llm_instance: BaseChatModel, tools: List[Any]):
        """Build one LangGraph workflow.

        中文说明: workflow = agent -> stage_check -> agent/finally_report/end。
        English: workflow = agent -> stage_check -> agent/finally_report/end.
        """
        finally_report = create_reporter_node(self.agent_config.get("reporter", {}), llm_instance, True)
        react_agent = create_react_agent(
            model=llm_instance,
            tools=tools,
            prompt=self.prompt_template.template,
            state_schema=CustomAgentState,
        )

        def should_continue(state: CustomAgentState) -> str:
            stage = StageEnum(state["current_stage"])
            if stage == StageEnum.MAX_ITERATIONS:
                raise ValueError("Maximum iterations reached.")
            if stage == StageEnum.DONE:
                if not state.get("final_response"):
                    return "finally_report"
                if state["final_response"].get("is_success") is False:
                    return "agent"
                return END
            if stage == StageEnum.ERROR:
                recent = state["messages"][-5:]
                context = "\n".join(f"- {type(message).__name__}: {getattr(message, 'content', '')}" for message in recent)
                raise ValueError(f"Agent entered error state. Recent messages:\n{context}")
            return "agent"

        graph = StateGraph(state_schema=CustomAgentState)
        graph.add_node("agent", react_agent)
        graph.add_node("stage_check", self.stage_check)
        graph.add_node("finally_report", finally_report)
        graph.set_entry_point("agent")
        graph.add_edge("agent", "stage_check")
        graph.add_edge("finally_report", "stage_check")
        graph.add_conditional_edges("stage_check", should_continue, {"agent": "agent", "finally_report": "finally_report", END: END})
        return graph.compile(debug=self.verbose, checkpointer=InMemorySaver())

    def get_input(self, image_path: str) -> dict[str, Any]:
        """Build initial image payload.

        中文说明: support_vision=true 时把图片 base64 放入首条 HumanMessage。
        English: When support_vision=true, image base64 is embedded in the first
        HumanMessage.
        """
        base64_str, _, image_format = self.image_manager.get_base64(
            src_image=image_path,
            is_resize=False,
            is_center_crop=True,
            target_width=128,
            target_height=128,
        )
        return {"image_base64": base64_str, "image_path": image_path, "image_format": image_format}

    def think_and_act(self, image_path: str, workflow_id: int) -> CustomAgentState:
        """Run one workflow for one image.

        中文说明: workflow_id 选择具体 LLM/工具实例, 单 LLM 配置固定为 0。
        English: workflow_id selects one LLM/tool instance; single-LLM configs
        always use 0.
        """
        origin_input = self.get_input(image_path)
        checkpoint_id = str(hash(f"{image_path}:{workflow_id}"))
        workflow_llm = self.llms[workflow_id]
        callback_handler = ForensicCallbackHandler(
            session_id=checkpoint_id,
            model_name=getattr(workflow_llm, "model", None) or getattr(workflow_llm, "model_name", None),
        )
        callback_handler.clear()
        config = {
            "configurable": {"thread_id": checkpoint_id, "checkpoint_id": checkpoint_id},
            "callbacks": [callback_handler],
            "recursion_limit": self.max_iterations * 2,
        }

        image_format = str(origin_input.get("image_format", "jpeg")).lower()
        if self.forensic_llm.support_vision:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{origin_input['image_base64']}"}},
                {"type": "text", "text": self.QUESTION_PROMPT},
            ]
            human_msg = HumanMessage(content=content)
        else:
            human_msg = HumanMessage(content=f"{self.QUESTION_PROMPT} The image is available to tools; rely on tool outputs.")

        input_state = CustomAgentState(
            messages=[human_msg],
            origin_input=origin_input,
            image_path=origin_input["image_path"],
            workflow_id=workflow_id,
            current_stage=StageEnum.INITIAL,
            iterations=0,
        )
        agent_state: dict = self.workflow_info[workflow_id].invoke(input_state, config=config, durability="sync")
        agent_state["metrics"] = callback_handler.get_summary()
        if agent_state.get("final_response"):
            agent_state["final_response"]["metrics"] = agent_state["metrics"]
        return agent_state

    def start_workers(self) -> None:
        """Start background workers.

        中文说明: 最小 CLI 暂不使用并发, 该方法保留给未来扩展。
        English: The minimal CLI does not use concurrency yet; this method is
        kept for future expansion.
        """
        if self._executor.is_running:
            logger.warning("Workflow workers are already running.")
            return
        self._executor.start()
        self._next_workflow = 0

    def shutdown_workers(self, wait: bool = True) -> None:
        """Stop background workers.

        中文说明: 允许重复调用, 未启动时直接返回。
        English: Repeated calls are allowed and return immediately if workers
        are not running.
        """
        if self._executor.is_running:
            self._executor.shutdown(wait=wait)

    def submit_task(self, image_path: str, result_queue: Queue, error_queue: Queue, workflow_id: int | None = None) -> int:
        """Submit one image to background workers.

        中文说明: 当前 test 走串行路径, 该接口不参与最小运行。
        English: Current test mode is serial, so this interface is not used in
        the minimal run.
        """
        if not self._executor.is_running:
            raise RuntimeError("Workflow workers are not running.")
        if workflow_id is None:
            with self._lock:
                workflow_id = self._next_workflow
                self._next_workflow = (self._next_workflow + 1) % len(self.llms)
        self._executor.submit(workflow_id, image_path, result_queue, error_queue)
        return workflow_id

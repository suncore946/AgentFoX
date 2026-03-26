import atexit
import json
from queue import Empty, Queue
import re
from threading import Lock, Thread
from loguru import logger
from typing import Callable, Dict, Any, List, Optional, Tuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel

from .forensic_llm import ForensicLLM
from .forensic_template import ForensicTemplate
from .forensic_tools import ForensicTools
from .forensic_reporter import create_reporter_node
from .forensic_callback_handler import ForensicCallbackHandler
from .agent_state import CustomAgentState, StageEnum
from ..manager.image_manager import ImageManager


class WorkflowExecutor:
    def __init__(self, worker_count: Dict[int, int], worker_fn: Callable[[int, Tuple[Any, ...]], None]):
        self._queues = {wid: Queue() for wid in worker_count}
        self._threads = {wid: [] for wid in worker_count}
        self._shutdown = False
        self._running = False
        self._worker_fn = worker_fn
        self._worker_count = worker_count

    def start(self):
        if self._running:
            return
        self._shutdown = False
        for wid, count in self._worker_count.items():
            for idx in range(count):
                t = Thread(target=self._run, name=f"Workflow-{wid}-Worker-{idx}", args=(wid,), daemon=True)
                t.start()
                self._threads[wid].append(t)
        self._running = True

    def shutdown(self, wait: bool = True):
        if not self._running:
            return
        self._shutdown = True
        for wid, q in self._queues.items():
            for _ in self._threads[wid]:
                q.put(None)
        if wait:
            for threads in self._threads.values():
                for t in threads:
                    t.join(timeout=5.0)
        for threads in self._threads.values():
            threads.clear()
        self._running = False

    def submit(self, workflow_id: int, *payload: Any):
        if not self._running or self._shutdown:
            raise RuntimeError("executors shutting down")
        queue = self._queues.get(workflow_id)
        if queue is None:
            raise KeyError(f"unknown workflow {workflow_id}")
        queue.put(payload)

    @property
    def is_running(self) -> bool:
        return self._running and not self._shutdown

    def _run(self, workflow_id: int):
        q = self._queues[workflow_id]
        logger.debug(f"workflow {workflow_id} worker started")
        while True:
            try:
                task = q.get(timeout=0.5)
            except Empty:
                if self._shutdown:
                    break
                continue
            try:
                if task is None:
                    break
                self._worker_fn(workflow_id, task)
            finally:
                q.task_done()
        logger.info(f"workflow {workflow_id} worker stopped")


class ForensicAgent:
    # Prior to the final reporting stage, analyze and summarize outputs from all stages.
    # Upon completing each stage, simply summarize and analyze the key outputs.
    # Only for stages where result conflicts arise, provide analysis and summary, with explicit explanation of the conflicts identified.
    QUESTION_PROMPT = """
Analyze the image comprehensively from multiple perspectives to determine its authenticity, and summarize all findings before generating the final report.
"""

    CHECK_STAGE_ERROR_MSG = """Unrecognized stage: Please analyze the current stage name. 
If you need to proceed to the next state, please reply with 'update stage to: [next state name]'. 
If the final report has already done, please add "update stage to: done" at the end. 
If tool is unavailable or an error occurs, please reply with "update stage to: error".
Do not add any extra symbols to the state name.
"""

    def __init__(
        self,
        config: dict,
        forensic_tool: ForensicTools,
        forensic_llm: ForensicLLM,
        image_manager: ImageManager,
        is_debug: bool = False,
    ):
        self.agent_config = config
        self.forensic_tool = forensic_tool
        self.forensic_llm = forensic_llm

        self.llms: List[BaseChatModel] = forensic_llm.llms
        self.callback_handler = ForensicCallbackHandler()
        self.image_manager = image_manager

        # 配置参数
        self.verbose = is_debug
        self.max_iterations = self.agent_config.get("max_iterations", 40)

        # 构建模板
        template = ForensicTemplate(self.agent_config)
        self.prompt_template: PromptTemplate = template.build_agent_template()

        # 构建执行图
        self.workflow_info: Dict[int, StateGraph] = {}
        self.num_workflows = len(self.llms)
        for i, llm in enumerate(self.llms):
            self.workflow_info[i] = self._build_graph(llm)

        if self.num_workflows == 0:
            raise ValueError("No valid workflow constructed, please check the configuration and LLMs.")
        else:
            self._workers_started = False
            self.per_workflow_workers: int = int(self.agent_config.get("per_workflow_workers", 1))

            worker_count = {i: self.per_workflow_workers for i in range(len(self.llms))}
            self._executor = WorkflowExecutor(worker_count, self._dispatch_task)

            self._next_workflow = 0
            atexit.register(lambda: self.shutdown_workers(wait=False))

        logger.info("智能AIGC检测取证Agent初始化完成")
        self._lock = Lock()  # 为线程安全添加锁

    def _dispatch_task(self, workflow_id: int, payload: Tuple[Any, ...]) -> None:
        image_path, result_queue, error_queue = payload
        try:
            result = self.think_and_act(image_path, workflow_id)
            result_queue.put((image_path, result))
        except Exception as exc:
            logger.error(f"Workflow {workflow_id} 执行失败{image_path}: {exc}")
            error_queue.put((image_path, exc))

    def stage_check(self, state: CustomAgentState) -> Dict[str, Any]:
        # 更新阶段检查
        def _check_stage(messages: List[AnyMessage]) -> StageEnum:
            """检查目前任务进度阶段。
            约定：Agent 在最后一条 AIMessage 中包含 'update stage to: XXXX'
            """

            available_stages = ", ".join(str(m.value) for m in StageEnum.__members__.values())
            available_tools = ", ".join(t.name for t in self.forensic_tool.get_all_tools())
            error_msg = self.CHECK_STAGE_ERROR_MSG + f"Available stages: {available_stages}." + f"Available toolkits: {available_tools}."

            if not messages:
                raise ValueError("消息为空，无法判断阶段")

            # 找到最后一个有效的 AIMessage(内容不为 None 或空字符串)
            ai_msg = None
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    content = m.content
                    # 检查内容是否有效
                    if content is not None:
                        if isinstance(content, str):
                            if content.strip():  # 非空字符串
                                ai_msg = m
                                break
                        else:  # dict 或 list 等其他类型,只要不为 None 就认为有效
                            ai_msg = m
                            break

            if ai_msg is None:
                logger.debug(f"未找到有效的 AIMessage，无法判断阶段: {state['image_path']}")
                raise ValueError(error_msg)

            reasoning = ai_msg.additional_kwargs.get("reasoning_content", "")
            content = ai_msg.content.lower()
            if isinstance(content, dict):
                # 你也可以在这里从结构化内容里取字段
                text = json.dumps(content, ensure_ascii=False)
            elif isinstance(content, list):
                # 比如一些模型会返回富文本结构
                text = json.dumps(content, ensure_ascii=False)
            else:
                text = str(content) if content is not None else ""
            low = reasoning + text.lower()

            # 解析 "update stage to: XXXX"
            stage_name: Optional[str] = None
            matches = re.findall(r"update stage to:\s*([a-zA-Z0-9_]+)", low)
            if matches:
                stage_name = matches[-1]  # 取最后一次匹配的结果

            if not stage_name:
                raise ValueError(error_msg)

            key = stage_name.upper()
            if key in StageEnum.__members__:
                return StageEnum[key]

            raise ValueError(error_msg)

        try:
            state["current_stage"] = _check_stage(state["messages"])
            if state["current_stage"] == StageEnum.FINALLY_REPORT:
                state["messages"].append(
                    HumanMessage(
                        content="""In this step, do not use any tools. Analyze and produce a report. And reply "update stage to: 'done'" in the end. """
                    )
                )
        except Exception as e:
            logger.debug(f"阶段解析失败: {e}")
            state["messages"].append(HumanMessage(content=str(e)))
        finally:
            state.pop("remaining_steps", None)  # 清除剩余步骤数，避免影响后续
            iterations: int = len(state["messages"]) // 2  # 每轮包含 Human + AI 两条消息
            if iterations >= self.max_iterations:
                state["current_stage"] = StageEnum.MAX_ITERATIONS
                state["messages"].append(AIMessage(content="已达到最大迭代次数，流程终止。"))
                return state
        return state

    def _build_graph(self, llm_instance: BaseChatModel):
        """构建简化的执行图"""
        tools = self.forensic_tool.get_all_tools()

        # 生成报告节点
        finally_report = create_reporter_node(self.agent_config.get("reporter", {}), llm_instance, True)

        # 构建预置 ReAct 代理为一个节点
        react_agent = create_react_agent(
            model=llm_instance,
            tools=tools,
            prompt=self.prompt_template.template,
            state_schema=CustomAgentState,
        )

        def should_continue(state: CustomAgentState) -> str:
            stage = StageEnum(state["current_stage"])
            if stage == StageEnum.MAX_ITERATIONS:
                raise ValueError("达到最大迭代次数，结束流程")
            elif stage == StageEnum.DONE:
                # return "output_check"
                if not state.get("final_response", None):
                    return "finally_report"
                if state["final_response"]["is_success"] is False:
                    return "agent"
                return END
            elif stage == StageEnum.ERROR:
                # 获取最后5条消息内容, 如果有5条, 没有则全部获取
                last_msgs = state["messages"][-5:]
                error_context = "\n".join([f"- {type(m).__name__}: {m.content}" for m in last_msgs if hasattr(m, "content")])
                raise ValueError(f"Agent encountered an error state, terminating workflow. Last messages:\n{error_context}")
            else:
                return "agent"

        # 构建图
        graph = StateGraph(state_schema=CustomAgentState)
        graph.add_node("agent", react_agent)
        graph.add_node("stage_check", self.stage_check)
        graph.add_node("finally_report", finally_report)

        graph.set_entry_point("agent")

        graph.add_edge("agent", "stage_check")
        graph.add_edge("finally_report", "stage_check")

        graph.add_conditional_edges(
            "stage_check",
            should_continue,
            {
                "agent": "agent",
                "finally_report": "finally_report",
                END: END,
            },
        )

        # 编译
        return graph.compile(debug=self.verbose, checkpointer=InMemorySaver())

    def get_input(self, image_path):
        base64_str, _, image_format = self.image_manager.get_base64(
            src_image=image_path,
            is_resize=False,
            is_center_crop=True,
            target_width=128,
            target_height=128,
        )
        return {
            "image_base64": base64_str,
            "image_path": image_path,
            "image_format": image_format,
        }

    def think_and_act(self, image_path, workflow_id: int) -> CustomAgentState:
        """执行单个 workflow"""
        origin_input = self.get_input(image_path)
        checkpoint_id = str(hash(str(image_path) + str(workflow_id)))
        logger.debug(f"开始执行 - checkpoint_id: {checkpoint_id}, workflow_id: {workflow_id}")

        config = {
            "configurable": {"thread_id": checkpoint_id, "checkpoint_id": checkpoint_id},
            # "callbacks": [self.callback_handler], # 如果需要详细的过程日志，可以开启此回调
            "recursion_limit": self.max_iterations * 2,
        }

        def _init_state(origin_input: Dict[str, Any]) -> CustomAgentState:
            image_base64 = origin_input.get("image_base64")
            image_format = origin_input.get("image_format").lower()
            if self.forensic_llm.support_vision:
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"},
                    },
                    {
                        "type": "text",
                        "text": f"{self.QUESTION_PROMPT}, Please examine the image’s authenticity and use available tools to deliver a definitive conclusion.",
                    },
                ]
                human_msg = HumanMessage(content=content)
            else:
                human_msg = HumanMessage(
                    content=f"{self.QUESTION_PROMPT}, The image has been provided to the tools for analysis. Rely on the results provided by the tools for your analysis.",
                )
            return CustomAgentState(
                messages=[human_msg],
                origin_input=origin_input,
                image_path=origin_input.get("image_path"),
                current_stage=StageEnum.INITIAL,
                iterations=0,
            )

        input_state = _init_state(origin_input)

        # 使用指定的 workflow
        agent_state: dict = self.workflow_info[workflow_id].invoke(
            input_state,
            config=config,
            durability="sync",
        )
        return agent_state

    def start_workers(self):
        """启动所有工作线程"""
        if self._executor.is_running:
            logger.warning("工作线程已经启动")
            return
        self._executor.start()
        total_workers = len(self.llms) * self.per_workflow_workers
        self._next_workflow = 0
        self._workers_started = True
        logger.info(f"发现共有{len(self.llms)}个LLM, 每个LLM启动{self.per_workflow_workers}个线程, 共启动 {total_workers} 个线程")

    def shutdown_workers(self, wait: bool = True):
        """关闭所有工作线程"""
        if not self._executor.is_running:
            return
        self._executor.shutdown(wait=wait)
        self._workers_started = False
        logger.info("所有工作线程已关闭")

    def submit_task(
        self,
        image_path: str,
        result_queue: Queue,
        error_queue: Queue,
        workflow_id: int = None,
    ) -> int:
        if not self._executor.is_running:
            raise RuntimeError("工作线程未启动，请先调用 start_workers()")

        if workflow_id is None:
            with self._lock:
                workflow_id = self._next_workflow
                self._next_workflow = (self._next_workflow + 1) % len(self.llms)

        self._executor.submit(workflow_id, image_path, result_queue, error_queue)
        return workflow_id

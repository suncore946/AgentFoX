from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union
from langchain_ollama import ChatOllama
from loguru import logger
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from ..utils import get_base_url


class ForensicLLM:
    def __init__(self, config: Union[Dict, List[Dict]]):
        self._llm_instances: List[BaseChatModel] = []
        target_hosts = config.pop("base_url")
        self.support_vision = config.pop("support_vision", False)
        if not isinstance(target_hosts, list):
            target_hosts = [target_hosts]

        if len(target_hosts) == 0:
            raise ValueError("At least one target_host must be specified in base_url")
        elif len(target_hosts) == 1:
            self._llm_instances.append(self._init_llm({**config, "base_url": target_hosts[0]}))
        else:
            # 并行初始化 LLM 实例，初始化完成后线程池自动关闭
            with ThreadPoolExecutor(max_workers=len(target_hosts)) as executor:
                # 为每个 host 构建独立的配置副本，避免并发修改同一 dict
                futures = [executor.submit(self._init_llm, {**config, "base_url": host}) for host in target_hosts]
                # 保持与 target_hosts 相同的顺序收集结果
                for fut in futures:
                    self._llm_instances.append(fut.result())

    @property
    def llm_num(self) -> int:
        return len(self._llm_instances)

    def get_pos_llm(self, pos: int) -> BaseChatModel:
        if pos < 0 or pos >= len(self._llm_instances):
            raise IndexError("LLM position out of range")
        return self._llm_instances[pos]

    def _init_llm(self, llm_config: Dict):
        """初始化单个LLM实例"""
        # 添加配置验证
        if not llm_config:
            raise ValueError("LLM configuration is required")

        model_provider = llm_config.get("model_provider", "").lower()
        model_name = llm_config.get("model")

        if not model_name:
            raise ValueError("model name is required in configuration")

        # 创建配置副本以避免修改原始配置
        config_copy = llm_config.copy()

        # 移除不属于 LLM 初始化的参数
        config_copy.pop("prompt_path", None)

        # 添加特定于提供商的参数
        if model_provider == "openai":
            config_copy["base_url"] = get_base_url(llm_config)
            llm_instance: BaseChatModel = init_chat_model(**config_copy)
        elif model_provider == "ollama":
            llm_instance: BaseChatModel = init_chat_model(**config_copy, validate_model_on_init=True)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

        # 检查工具调用支持
        try:
            self.check_tool_callback(llm_instance, model_name)
        except Exception as e:
            logger.error(f"Tool callback check failed for {model_provider}/{config_copy.get('base_url')}/{model_name}: {e}")
            raise e

        logger.info(f"LLM initialized: {type(llm_instance).__name__} ({model_provider}/{config_copy["base_url"]}/{model_name})")
        return llm_instance

    @property
    def llms(self) -> List[BaseChatModel]:
        return self._llm_instances

    @property
    def llm(self) -> BaseChatModel:
        if len(self._llm_instances) < 1:
            raise ValueError("No LLM instances available")
        return self._llm_instances[0]

    def check_tool_callback(self, llm_instance: BaseChatModel, model_name: str) -> bool:
        """检查LLM是否支持工具调用"""
        bind_method = getattr(llm_instance, "bind_tools", None)
        if not callable(bind_method):
            logger.warning(f"Model {model_name} does not expose bind_tools")
            return False
        logger.info(f"Model {model_name} declares tool calling support")
        return True

"""LLM factory for AgentFoX.

中文说明: 只保留 OpenAI-compatible 与 Ollama 两类常见推理后端, 不内置私有地址或密钥。
English: Only OpenAI-compatible and Ollama backends are kept, with no private
hosts or keys embedded.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from loguru import logger

from ..utils import get_base_url


class ForensicLLM:
    """Create one or more chat model instances.

    中文说明: base_url 可以是字符串或列表; 列表会创建多个 workflow。
    English: base_url may be a string or list; a list creates multiple workflows.
    """

    def __init__(self, config: Dict):
        if not isinstance(config, dict):
            raise TypeError("llm config must be a dictionary.")
        config_copy = dict(config)
        target_hosts = config_copy.pop("base_url", None)
        if not target_hosts:
            raise ValueError("llm.base_url is required.")
        self.support_vision = bool(config_copy.pop("support_vision", False))
        hosts = target_hosts if isinstance(target_hosts, list) else [target_hosts]
        if not hosts:
            raise ValueError("At least one llm.base_url must be configured.")

        self._llm_instances: List[BaseChatModel] = []
        if len(hosts) == 1:
            self._llm_instances.append(self._init_llm({**config_copy, "base_url": hosts[0]}))
        else:
            with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
                futures = [executor.submit(self._init_llm, {**config_copy, "base_url": host}) for host in hosts]
                self._llm_instances = [future.result() for future in futures]

    @property
    def llm_num(self) -> int:
        """Return the number of configured LLM instances.

        中文说明: batch 并发可按该数量分配 workflow。
        English: Batch concurrency can distribute workflows across this count.
        """
        return len(self._llm_instances)

    @property
    def llms(self) -> List[BaseChatModel]:
        """Return all LLM instances.

        中文说明: ForensicAgent 会为每个 LLM 构建独立 workflow。
        English: ForensicAgent builds one workflow per LLM.
        """
        return self._llm_instances

    def get_pos_llm(self, pos: int) -> BaseChatModel:
        """Return one LLM by position.

        中文说明: 位置越界立即报错, 避免任务被提交到不存在的 workflow。
        English: Out-of-range positions fail fast so tasks are not submitted to
        missing workflows.
        """
        if pos < 0 or pos >= len(self._llm_instances):
            raise IndexError("LLM position out of range.")
        return self._llm_instances[pos]

    def _init_llm(self, llm_config: Dict) -> BaseChatModel:
        """Initialize one chat model.

        中文说明: OpenAI-compatible 后端从环境变量或配置读取 API key, 配置模板不写密钥。
        English: OpenAI-compatible backends read API keys from environment or
        config, while the template never contains a key.
        """
        model_provider = str(llm_config.get("model_provider", "openai")).lower()
        model_name = llm_config.get("model")
        if not model_name:
            raise ValueError("llm.model is required.")

        config_copy = dict(llm_config)
        config_copy.pop("prompt_path", None)
        if model_provider == "openai":
            config_copy["base_url"] = get_base_url(config_copy)
            if not config_copy.get("api_key"):
                config_copy["api_key"] = os.environ.get("OPENAI_API_KEY")
            if not config_copy.get("api_key"):
                raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible providers.")
            llm_instance = init_chat_model(**config_copy)
        elif model_provider == "ollama":
            llm_instance = init_chat_model(**config_copy, validate_model_on_init=True)
        else:
            raise ValueError(f"Unsupported llm.model_provider: {model_provider}")

        self.check_tool_callback(llm_instance, model_name)
        logger.info(f"LLM initialized: {type(llm_instance).__name__} ({model_provider}/{config_copy['base_url']}/{model_name})")
        return llm_instance

    def check_tool_callback(self, llm_instance: BaseChatModel, model_name: str) -> bool:
        """Check whether the model exposes tool binding.

        中文说明: 不强制失败, 但会记录警告; 某些模型声明不完整但仍可运行。
        English: This does not fail hard; some models have incomplete metadata
        but still work.
        """
        bind_method = getattr(llm_instance, "bind_tools", None)
        if not callable(bind_method):
            logger.warning(f"Model {model_name} does not expose bind_tools.")
            return False
        logger.info(f"Model {model_name} declares tool calling support.")
        return True

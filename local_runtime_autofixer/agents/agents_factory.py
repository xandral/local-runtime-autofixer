import os
import logging
from enum import Enum
from typing import Optional, Dict, List

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from local_runtime_autofixer.agents.tools import get_function_details
from local_runtime_autofixer.agents.prompts import (
    guardian_prompt,
    incident_responder_prompt,
)

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AgentFactory")


class ModelProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    LOCAL = "local"


class ModelConfig:
    def __init__(
        self,
        provider: ModelProvider,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def from_env(cls, provider: ModelProvider) -> "ModelConfig":
        if provider == ModelProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is missing")
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return cls(provider, model_name, api_key)

        elif provider == ModelProvider.CLAUDE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is missing")
            model_name = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-latest")
            return cls(provider, model_name, api_key)

        elif provider == ModelProvider.LOCAL:
            model_name = os.getenv("LOCAL_MODEL")
            if not model_name:
                raise ValueError("LOCAL_MODEL environment variable is missing")
            base_url = os.getenv("LOCAL_BASE_URL")
            if not base_url:
                raise ValueError("LOCAL_BASE_URL environment variable is missing")
            return cls(provider, model_name, None, base_url)

        else:
            raise ValueError(f"Unsupported provider: {provider}")


class LLMAgentWrapper:
    def __init__(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        tools: Optional[List] = None,
        extra_model_kwargs: Optional[Dict] = None,
    ):
        self.cfg = model_cfg
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.extra_model_kwargs = extra_model_kwargs or {}

        if self.cfg.provider == ModelProvider.OPENAI:
            self.model = ChatOpenAI(
                model=self.cfg.model_name,
                api_key=self.cfg.api_key,
                base_url=self.cfg.base_url,
                **self.extra_model_kwargs,
            )
        elif self.cfg.provider == ModelProvider.CLAUDE:
            self.model = ChatAnthropic(
                model=self.cfg.model_name,
                api_key=self.cfg.api_key,
                base_url=self.cfg.base_url,
                **self.extra_model_kwargs,
            )
        elif self.cfg.provider == ModelProvider.LOCAL:
            self.model = ChatOllama(
                model=self.cfg.model_name,
                base_url=self.cfg.base_url,
                **self.extra_model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.cfg.provider}")

        self._agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    async def run(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = await self._agent.ainvoke({"messages": messages})
        return response["messages"][-1].content


class AgentFactory:
    def __init__(
        self,
        agent_configs: Optional[Dict[str, Dict]] = None,
        provider_priority: Optional[List[ModelProvider]] = None,
    ):
        self.provider_priority = provider_priority or [
            ModelProvider.OPENAI,
            ModelProvider.CLAUDE,
            ModelProvider.LOCAL,
        ]

        self._standard_prompts = {
            "fixer": incident_responder_prompt,
            "guardian": guardian_prompt,
        }

        self._default_agent_configs = {
            "fixer": {
                "prompt_key": "fixer",
                "tools": [get_function_details],
                "extra_model_kwargs": {"temperature": 0.0},
            },
            "guardian": {
                "prompt_key": "guardian",
                "tools": [],
                "extra_model_kwargs": {"temperature": 0.0},
            },
        }

        self.agent_configs = agent_configs or self._default_agent_configs
        self.agents: Dict[str, LLMAgentWrapper] = {}

        self._setup_agents()

    def _select_default_provider(self) -> ModelConfig:
        for prov in self.provider_priority:
            try:
                cfg = ModelConfig.from_env(prov)
                logger.info(f"Selected default provider: {prov.value}")
                return cfg
            except Exception as e:
                logger.warning(f"Unable to configure {prov.value}: {e}")
        raise RuntimeError("No valid LLM provider found in environment variables.")

    def _setup_agents(self):
        default_cfg = self._select_default_provider()

        for agent_name, config in self.agent_configs.items():
            if "provider" in config:
                provider = config["provider"]
                model_cfg = ModelConfig.from_env(provider)
            else:
                model_cfg = default_cfg

            if "system_prompt" in config:
                system_prompt = config["system_prompt"]
            else:
                if agent_name in self._standard_prompts:
                    prompt_dict = self._standard_prompts[agent_name]
                    if model_cfg.provider.value not in prompt_dict:
                        raise ValueError(
                            f"Missing prompt for provider {model_cfg.provider.value} in agent {agent_name}"
                        )
                    system_prompt = prompt_dict[model_cfg.provider.value]
                else:
                    raise ValueError(
                        f"Missing prompt_key or system_prompt for agent: {agent_name}"
                    )

            tools = config.get("tools", []).copy()
            if agent_name == "fixer" and get_function_details not in tools:
                tools.append(get_function_details)

            extra_kwargs = config.get("extra_model_kwargs", {})

            wrapper = LLMAgentWrapper(
                model_cfg=model_cfg,
                system_prompt=system_prompt,
                tools=tools,
                extra_model_kwargs=extra_kwargs,
            )

            self.agents[agent_name] = wrapper

    def __getattr__(self, name: str):
        if name.endswith("_agent"):
            agent_key = name.replace("_agent", "")
            if agent_key in self.agents:
                return self.agents[agent_key]
        raise AttributeError(f"No agent named '{name}' found")

import os
import pytest
from enum import Enum

from local_runtime_autofixer.agents.agents_factory import (
    ModelConfig,
    ModelProvider,
    AgentFactory,
    LLMAgentWrapper,
)

# ------------------------------------------------------------------------------
# TEST ModelConfig.from_env
# ------------------------------------------------------------------------------


class DummyProvider(Enum):
    """Dummy provider to verify ValueError on unknown provider."""

    UNKNOWN = "unknown"


def test_from_env_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model-01")
    cfg = ModelConfig.from_env(ModelProvider.OPENAI)

    assert cfg.provider is ModelProvider.OPENAI
    assert cfg.api_key == "test_openai_key"
    assert cfg.model_name == "test-model-01"


def test_from_env_openai_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError) as exc:
        ModelConfig.from_env(ModelProvider.OPENAI)
    assert "OPENAI_API_KEY environment variable is missing" in str(exc.value)


def test_from_env_openai_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_2")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    cfg = ModelConfig.from_env(ModelProvider.OPENAI)

    assert cfg.provider is ModelProvider.OPENAI
    assert cfg.api_key == "test_key_2"
    assert cfg.model_name == "gpt-4o-mini"


def test_from_env_claude_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude_key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-test-42")
    cfg = ModelConfig.from_env(ModelProvider.CLAUDE)

    assert cfg.provider is ModelProvider.CLAUDE
    assert cfg.api_key == "claude_key"
    assert cfg.model_name == "claude-test-42"


def test_from_env_claude_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    with pytest.raises(ValueError) as exc:
        ModelConfig.from_env(ModelProvider.CLAUDE)
    assert "ANTHROPIC_API_KEY environment variable is missing" in str(exc.value)


def test_from_env_claude_default_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic_key_2")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    cfg = ModelConfig.from_env(ModelProvider.CLAUDE)

    assert cfg.provider is ModelProvider.CLAUDE
    assert cfg.api_key == "anthropic_key_2"
    assert cfg.model_name == "claude-3-7-sonnet-latest"


def test_from_env_local_success(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    monkeypatch.setenv("LOCAL_MODEL", "local-model-9000")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8080")
    cfg = ModelConfig.from_env(ModelProvider.LOCAL)

    assert cfg.provider is ModelProvider.LOCAL
    assert cfg.api_key is None
    assert cfg.model_name == "local-model-9000"
    assert cfg.base_url == "http://localhost:8080"


def test_from_env_local_missing_model(monkeypatch):
    monkeypatch.delenv("LOCAL_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:9001")
    with pytest.raises(ValueError) as exc:
        ModelConfig.from_env(ModelProvider.LOCAL)
    assert "LOCAL_MODEL environment variable is missing" in str(exc.value)


def test_from_env_local_missing_base_url(monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL", "foo")
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    with pytest.raises(ValueError) as exc:
        ModelConfig.from_env(ModelProvider.LOCAL)
    assert "LOCAL_BASE_URL environment variable is missing" in str(exc.value)


def test_from_env_unsupported_provider():
    with pytest.raises(ValueError) as exc:
        ModelConfig.from_env(DummyProvider.UNKNOWN)  # type: ignore
    assert "Unsupported provider" in str(exc.value)


# ------------------------------------------------------------------------------
# TEST LLMAgentWrapper (basic attribute assignment)
# ------------------------------------------------------------------------------


class DummyModelConfig:
    """Dummy ModelConfig to bypass actual ChatOpenAI/ChatAnthropic/ChatOllama."""

    def __init__(self):
        self.provider = ModelProvider.LOCAL
        self.model_name = "dummy"
        self.api_key = None
        self.base_url = "http://dummy"


class DummyModel:
    """Stub to create model instance without real credentials/API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_llm_agent_wrapper_assignment(monkeypatch):
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOpenAI",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatAnthropic",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOllama",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.create_react_agent",
        lambda model, tools, prompt: object(),
    )

    dummy_cfg = DummyModelConfig()
    wrapper = LLMAgentWrapper(
        model_cfg=dummy_cfg,
        system_prompt="TEST_PROMPT",
        tools=["foo"],
        extra_model_kwargs={"x": 1},
    )

    assert wrapper.cfg is dummy_cfg
    assert wrapper.system_prompt == "TEST_PROMPT"
    assert wrapper.tools == ["foo"]
    assert hasattr(wrapper, "_agent")
    assert isinstance(wrapper.model, DummyModel)
    assert wrapper.model.kwargs["model"] == "dummy"
    assert wrapper.model.kwargs["base_url"] == "http://dummy"
    assert wrapper.model.kwargs["x"] == 1


# ------------------------------------------------------------------------------
# TEST AgentFactory (fallback and agent creation)
# ------------------------------------------------------------------------------


def test_agent_factory_no_env(monkeypatch):
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
        "LOCAL_MODEL",
        "LOCAL_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(RuntimeError) as exc:
        AgentFactory(
            provider_priority=[
                ModelProvider.OPENAI,
                ModelProvider.CLAUDE,
                ModelProvider.LOCAL,
            ]
        )
    assert "No valid LLM provider" in str(exc.value)


def test_agent_factory_prefers_openai_over_other(monkeypatch):
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
        "LOCAL_MODEL",
        "LOCAL_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "oi_key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth_key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-test")
    monkeypatch.setenv("LOCAL_MODEL", "local-test")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost")

    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOpenAI",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatAnthropic",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOllama",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.create_react_agent",
        lambda model, tools, prompt: object(),
    )

    factory = AgentFactory()

    assert factory.fixer_agent is not None
    assert factory.guardian_agent is not None
    assert factory.fixer_agent.cfg.provider is ModelProvider.OPENAI
    assert factory.guardian_agent.cfg.provider is ModelProvider.OPENAI


def test_agent_factory_fallback_to_claude(monkeypatch):
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
        "LOCAL_MODEL",
        "LOCAL_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthro_key_2")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-foo")
    monkeypatch.setenv("LOCAL_MODEL", "local-bar")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:1234")

    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOpenAI",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatAnthropic",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOllama",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.create_react_agent",
        lambda model, tools, prompt: object(),
    )

    factory = AgentFactory(
        provider_priority=[
            ModelProvider.OPENAI,
            ModelProvider.CLAUDE,
            ModelProvider.LOCAL,
        ]
    )

    assert factory.fixer_agent is not None
    assert factory.fixer_agent.cfg.provider is ModelProvider.CLAUDE
    assert factory.guardian_agent.cfg.provider is ModelProvider.CLAUDE


def test_agent_factory_fallback_to_local(monkeypatch):
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("LOCAL_MODEL", "local-xyz")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://loc:9999")

    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOpenAI",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatAnthropic",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOllama",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.create_react_agent",
        lambda model, tools, prompt: object(),
    )

    factory = AgentFactory(
        provider_priority=[
            ModelProvider.OPENAI,
            ModelProvider.CLAUDE,
            ModelProvider.LOCAL,
        ]
    )

    assert factory.fixer_agent is not None
    assert factory.fixer_agent.cfg.provider is ModelProvider.LOCAL
    assert factory.guardian_agent.cfg.provider is ModelProvider.LOCAL


def test_agent_factory_custom_agent_configs(monkeypatch):
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("LOCAL_MODEL", "local-abc")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:5678")

    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.ChatOllama",
        lambda **kwargs: DummyModel(**kwargs),
    )
    monkeypatch.setattr(
        "local_runtime_autofixer.agents.agents_factory.create_react_agent",
        lambda model, tools, prompt: object(),
    )

    custom_agent_configs = {
        "fixer": {
            "prompt_key": "fixer",  # 🔁 Fixed this
            "tools": [],
            "extra_model_kwargs": {"temperature": 0.5},
        },
        "guardian": {
            "prompt_key": "guardian",  # 🔁 Fixed this
            "tools": [],
            "extra_model_kwargs": {"temperature": 0.1},
        },
    }

    factory = AgentFactory(
        provider_priority=[ModelProvider.LOCAL],
        agent_configs=custom_agent_configs,
    )

    assert factory.fixer_agent is not None
    assert factory.guardian_agent is not None

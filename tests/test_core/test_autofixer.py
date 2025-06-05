import pytest

from typing import Any, Tuple

from local_runtime_autofixer.utils.custom_exceptions import IncidentException
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler
from local_runtime_autofixer.agents.agents_factory import AgentFactory
from local_runtime_autofixer.agents.prompts import (
    GUARDIAN_TASK,
    INCIDENT_RESPONDER_PROMPT,
)

from local_runtime_autofixer.autofixer import LocalIncidentResponder


class DummyIncidentModel:
    """
    “Modello” di incidente semplificato.
    Contiene tutti i campi usati da LocalIncidentResponder._build_incident
    e può essere aggiornato da update_incident_with_response.
    """

    def __init__(self):
        self.id = "INC123"
        self.source_code = "def buggy():\n    1/0"
        self.module_path = "/path/to/fake_module.py"
        self.docstring = ""
        self.error_type = "ZeroDivisionError"
        self.error = "division by zero"
        self.traceback = (
            "Traceback (most recent call last):  … ZeroDivisionError: division by zero"
        )
        self.args = []
        self.kwargs = {}
        self.fix_applied = False
        self.fixed_output = None
        self.execution_error = None

    def model_dump(self) -> dict:
        """
        Restituisce un dict con campi come li si aspetta nel flusso di LocalIncidentResponder.
        """
        return {
            "id": self.id,
            "source_code": self.source_code,
            "module_path": self.module_path,
            "docstring": self.docstring,
            "error_type": self.error_type,
            "error": self.error,
            "traceback": self.traceback,
            "args": self.args,
            "kwargs": self.kwargs,
        }


class DummyIncidentHandler(BaseIncidentHandler):
    """
    Handler di incidente “fake” che registra le chiamate:
      - make_incident: restituisce sempre lo stesso DummyIncidentModel
      - update_incident_with_response: imposta i campi nel modello e lo restituisce
      - notify_autofix_incident: memorizza il modello notificato
    """

    def __init__(self):
        super().__init__()
        self.last_incident_model = None
        self.notified = False

    def make_incident(
        self,
        func: Any,
        module_path: str,
        args: tuple,
        kwargs: dict,
        task: str,
        exc: Exception,
        input_type: str,
        output_type: str,
        tb_str: str,
    ) -> DummyIncidentModel:
        incident = DummyIncidentModel()
        incident.args = list(args)
        incident.kwargs = kwargs.copy()
        self.last_incident_model = incident
        return incident

    def update_incident_with_response(
        self,
        incident_model: DummyIncidentModel,
        fixed_code: str,
        fixed_output: Any,
        execution_error: str | None,
        fix_applied: bool,
    ) -> DummyIncidentModel:
        incident_model.fixed_output = fixed_output
        incident_model.execution_error = execution_error
        incident_model.fix_applied = fix_applied
        self.last_incident_model = incident_model
        return incident_model

    def notify_autofix_incident(self, incident_model: DummyIncidentModel):
        self.notified = True
        self.last_notified = incident_model


class DummyAgent:
    """
    Simula un agent LLM con un metodo run() che restituisce sempre un valore prefissato.
    """

    def __init__(self, to_return: Any):
        self.to_return = to_return
        self.last_prompt = None

    async def run(self, prompt: str) -> Any:
        self.last_prompt = prompt
        return self.to_return


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """
    Prima di ciascun test, rimuovo eventuali variabili d’ambiente che potrebbero
    influire sulla creazione di AgentFactory (OPENAI_API_KEY, ecc.).
    """
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
        "LOCAL_MODEL",
        "LOCAL_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield


def test_auto_fix_sync_success(tmp_path, monkeypatch):

    handler = DummyIncidentHandler()

    dummy_fixer = DummyAgent("```python\ndef buggy():\n    return 42\n```")
    dummy_guardian = DummyAgent('{"verdict": true}')

    class DummyAgentFactory(AgentFactory):
        def __init__(self):

            self.fixer_agent = dummy_fixer
            self.guardian_agent = dummy_guardian

    class TestResponder(LocalIncidentResponder):
        def __init__(self):

            super().__init__(
                incident_handler=handler, agent_factory=DummyAgentFactory()
            )

        def _build_incident(
            self,
            func: Any,
            exc: Exception,
            args: tuple,
            kwargs: dict,
            serialized_input_type: str,
            serialized_output_type: str,
        ) -> DummyIncidentModel:

            return handler.make_incident(
                func,
                "/fake/path/module.py",
                args,
                kwargs,
                "auto_fix",
                exc,
                serialized_input_type,
                serialized_output_type,
                "traceback str",
            )

        @staticmethod
        async def _execute_fix_in_current_env(
            fixed_code: str, original_func: Any, args: tuple, kwargs: dict
        ) -> Tuple[Any, None]:

            return 999, None

    responder = TestResponder()

    @responder.auto_fix()
    def buggy_fn(x: int):
        return x / 0

    result = buggy_fn(5)

    assert result == 999

    assert isinstance(handler.last_incident_model, DummyIncidentModel)

    assert handler.last_incident_model.fixed_output == 999

    assert handler.notified is True

    assert INCIDENT_RESPONDER_PROMPT.split("{")[0] in dummy_fixer.last_prompt

    assert GUARDIAN_TASK.split("{")[0] in dummy_guardian.last_prompt


@pytest.mark.asyncio
async def test_auto_fix_async_success(monkeypatch):
    """
    Stessa logica del test precedente, ma su una coroutine async.
    """

    handler = DummyIncidentHandler()
    dummy_fixer = DummyAgent("```python\nasync def buggy():\n    return 'ok'\n```")
    dummy_guardian = DummyAgent('{"verdict": true}')

    class DummyAgentFactory(AgentFactory):
        def __init__(self):
            self.fixer_agent = dummy_fixer
            self.guardian_agent = dummy_guardian

    class TestResponder(LocalIncidentResponder):
        def __init__(self):
            super().__init__(
                incident_handler=handler, agent_factory=DummyAgentFactory()
            )

        def _build_incident(
            self,
            func: Any,
            exc: Exception,
            args: tuple,
            kwargs: dict,
            serialized_input_type: str,
            serialized_output_type: str,
        ) -> DummyIncidentModel:
            return handler.make_incident(
                func,
                "/fake/path/async_module.py",
                args,
                kwargs,
                "auto_fix",
                exc,
                serialized_input_type,
                serialized_output_type,
                "traceback str",
            )

        @staticmethod
        async def _execute_fix_in_current_env(
            fixed_code: str, original_func: Any, args: tuple, kwargs: dict
        ) -> Tuple[Any, None]:

            return "patched result", None

    responder = TestResponder()

    @responder.auto_fix()
    async def buggy_async(x: int):

        raise TypeError("dummy async error")

    out = await buggy_async(10)
    assert out == "patched result"
    assert handler.last_incident_model.fixed_output == "patched result"
    assert handler.notified is True

    assert INCIDENT_RESPONDER_PROMPT.split("{")[0] in dummy_fixer.last_prompt
    assert GUARDIAN_TASK.split("{")[0] in dummy_guardian.last_prompt


def test_auto_fix_guardian_reject_raises(monkeypatch):
    handler = DummyIncidentHandler()

    dummy_fixer = DummyAgent("```python\ndef buggy():\n    pass\n```")

    dummy_guardian = DummyAgent('{"verdict": false}')

    class DummyAgentFactory(AgentFactory):
        def __init__(self):
            self.fixer_agent = dummy_fixer
            self.guardian_agent = dummy_guardian

    class TestResponder(LocalIncidentResponder):
        def __init__(self):
            super().__init__(
                incident_handler=handler, agent_factory=DummyAgentFactory()
            )

        def _build_incident(
            self,
            func: Any,
            exc: Exception,
            args: tuple,
            kwargs: dict,
            serialized_input_type: str,
            serialized_output_type: str,
        ) -> DummyIncidentModel:
            return handler.make_incident(
                func,
                "/fake/path/module_reject.py",
                args,
                kwargs,
                "auto_fix",
                exc,
                serialized_input_type,
                serialized_output_type,
                "traceback str",
            )

        @staticmethod
        async def _execute_fix_in_current_env(
            fixed_code: str, original_func: Any, args: tuple, kwargs: dict
        ) -> Tuple[Any, None]:

            return None, None

    responder = TestResponder()

    @responder.auto_fix(max_retries=1)
    def always_error():
        raise ValueError("qualcosa viene male")

    with pytest.raises(IncidentException):
        always_error()

from local_runtime_autofixer.autofixer import LocalIncidentResponder
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler
from pydantic import BaseModel

incindent_handler = BaseIncidentHandler()


import os
import asyncio
from local_runtime_autofixer.agents.agents_factory import AgentFactory, ModelProvider


# Configurazione personalizzata degli agenti: aggiungiamo anche un agente "diff" e un agente "extra"
custom_agent_configs = {
    "fixer": {
        "provider": ModelProvider.OPENAI,
        "extra_model_kwargs": {"temperature": 0.1, "max_tokens": 500},
    },
    "guardian": {
        "provider": ModelProvider.LOCAL,
        "system_prompt": "Check if the code is correct",
        "tools": [],
        "extra_model_kwargs": {"temperature": 0.0},
    },
}
# http://localhost:11434
# Instantiate the AgentFactory with your custom config
agent_factory = AgentFactory(agent_configs=custom_agent_configs)


class InputModel(BaseModel):
    a: int
    b: int


class OutputModel(BaseModel):
    result: int


responder = LocalIncidentResponder(incindent_handler, agent_factory=agent_factory)


@responder.auto_fix(input_type=InputModel, output_type=int, notify=True, max_retries=1)
def faulty(a, b):
    # 'c' is undefined, triggers NameError
    return a + c


# _, r = asyncio.run(faulty(1, 2))
# print("RESULT AFTER FIX", r)
faulty(2, 3)

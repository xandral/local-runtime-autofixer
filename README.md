# LocalIncidentResponder

`LocalIncidentResponder` is a Python library that provides automatic runtime error detection, diagnosis, and correction for functions. It leverages Large Language Models (LLMs) to generate corrective patches and validate them before execution.

**Note:** This library is intended for local testing purposes only and is still under development. It should not be used in production environments due to potential security risks.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Key Features

- **`auto_fix` decorator**: intercepts exceptions in both synchronous and asynchronous functions.
- **Automatic patch generation** using a dedicated LLM fixer agent.
- **Patch validation** through a guardian agent before applying fixes.
- **Safe execution of patches** within the local environment.
- **Detailed logging and notification support**.
- **Support for Pydantic models** to define input/output schemas.
- Automatic writing of patch files with applied diffs.
- Configurable retry mechanism for patch generation attempts.

---

## Installation
Clone the repo and on a virtual env run `pip install -r requirements.txt`, it will install the current used version of `uv`.
To run the examples in the MAIN directory:
1. `export PYTHONPATH=$PWD`
2. `uv run examples/...`
To run pytests: `uv run pytest`


Moreover, is possible to install it as a package: run `uv build` to get the `.whl` file and use it to install it as a pip package. Up to now is not already published on pipy.

---

## Usage Overview

Decorate your functions with the `auto_fix` decorator to enable automatic error detection and patching. The decorator supports options for input/output schema validation, notification, retry limits, custom context, execution control, and patch validation.

---

## API

### `auto_fix(...)`

Decorator that enables auto-fixing of exceptions occurring in the target function.

**Parameters:**

- `input_type` (Pydantic model or Python type): Defines the expected input schema.
- `output_type` (Pydantic model or Python type): Defines the expected output schema.
- `notify` (bool): If true, sends notifications upon auto-fix events.
- `max_retries` (int): Maximum attempts to generate and apply a valid patch.
- `context` (str): Additional textual context passed to the LLM agents.
- `run_fixed_code` (bool): Whether to execute the patched code locally.
- `guardrail` (bool): Enables patch validation through the guardian agent.

---
# AutoFixer: Automated Runtime Incident Detection and Resolution

AutoFixer is a Python framework designed to detect runtime incidents (exceptions, errors) in your code, automatically generate fixes using AI-powered agents, validate patches, and optionally apply and notify about these fixes in real time. It will create a new file with the fix applied near module patched, to inspect in local the changes.


---

## Incident Handler Overview

The **Incident Handler** is a core component responsible for managing runtime incidents detected during function execution.

### Responsibilities

- Capture incident details such as function name, inputs, errors, and tracebacks.
- Create structured `Incident` records with all relevant metadata.
- Update incidents with fix attempts, including source patches and execution results.
- Notify about auto-fix events via logging or other channels.

### Extending with Subclassing

`BaseIncidentHandler` provides a default implementation designed to be extended. You can subclass it to:

- Persist incidents in databases or external systems.
- Integrate with custom notification channels like Slack, email, or monitoring dashboards.
- Add extra metadata (environment, user context).
- Customize logging formats and destinations.
- Implement specialized workflows, e.g., for security incidents.

### Example Custom Incident Handler

```python
class CustomIncidentHandler(BaseIncidentHandler):
    def notify_autofix_incident(self, incident: Incident):
        super().notify_autofix_incident(incident)
        # Custom notification logic
        send_slack_message(f"Auto-fix applied for incident {incident.id}")

    def save_incident_to_db(self, incident: Incident):
        # Custom persistence logic
        db.insert(incident.model_dump())
```


---

# LLM Agent Configuration Guide

This guide explains how to configure custom agents and model providers using the `AgentFactory` class.

## Environment Variables

Before using the agents, you need to set up environment variables for your chosen model providers:

### OpenAI Configuration
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini
```

### Anthropic Claude Configuration
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export CLAUDE_MODEL="claude-3-7-sonnet-latest"  # Optional, defaults to claude-3-7-sonnet-latest
```

### Local Model Configuration
```bash
export LOCAL_MODEL="your-local-model-name"
export LOCAL_BASE_URL="http://localhost:11434"  # Your local model endpoint
```

**NB**: The guardian agent has to return json with a boolean field called `verdict` to be used.

## Basic Usage

### Using Default Configuration
```python
from your_module import AgentFactory

# Create factory with default agents (fixer and guardian)
factory = AgentFactory()

# Access agents
fixer_response = await factory.fixer_agent.run("Fix this issue...")
guardian_response = await factory.guardian_agent.run("Check this content...")
```

### Specifying Provider Priority
```python
from your_module import AgentFactory, ModelProvider

# Prefer Claude over OpenAI over Local
factory = AgentFactory(
    provider_priority=[
        ModelProvider.CLAUDE,
        ModelProvider.OPENAI,
        ModelProvider.LOCAL
    ]
)
```

## Custom Agent Configuration

### Agent Configuration Structure

Each agent configuration is a dictionary with the following optional fields:

```python
agent_config = {
    "provider": ModelProvider.CLAUDE,           # Optional: specific provider for this agent
    "system_prompt": "Your custom prompt...",   # Optional: custom system prompt
    "tools": [tool1, tool2],                   # Optional: list of tools for the agent
    "extra_model_kwargs": {                    # Optional: additional model parameters
        "temperature": 0.7,
        "max_tokens": 1000
    }
}
```

### Configuration Fields Explained

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `provider` | `ModelProvider` | Specific model provider for this agent | Uses factory default |
| `system_prompt` | `str` | Custom system prompt text | Uses `prompt_key` lookup |
| `tools` | `List` | List of tool functions available to agent | `[]` |
| `extra_model_kwargs` | `Dict` | Additional parameters passed to the model | `{}` |

### Custom Configuration Examples


```python
from local_runtime_autofixer.autofixer import LocalIncidentResponder
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler
from pydantic import BaseModel

import os
import asyncio
from local_runtime_autofixer.agents.agents_factory import AgentFactory, ModelProvider

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

faulty(2,3)
```

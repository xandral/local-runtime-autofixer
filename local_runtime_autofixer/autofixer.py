import asyncio
import functools
import importlib
import inspect
import logging
import traceback
from typing import Any, Callable, Optional, Tuple, Type

import black
from pydantic import BaseModel

from local_runtime_autofixer.agents.agents_factory import AgentFactory
from local_runtime_autofixer.agents.prompts import (
    GUARDIAN_TASK,
    INCIDENT_RESPONDER_PROMPT,
)
from local_runtime_autofixer.utils.custom_exceptions import IncidentException
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler
from local_runtime_autofixer.utils.miscellaneous import extract_formatted_text

logger = logging.getLogger("AutoFixer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class LocalIncidentResponder:
    def __init__(
        self,
        incident_handler: Optional[BaseIncidentHandler] = None,
        global_context: Optional[str] = "",
        agent_factory: Optional[AgentFactory] = None,
    ):
        self.global_context = global_context
        self.incident_handler = incident_handler or BaseIncidentHandler()
        self.agent_factory = agent_factory or AgentFactory()

    def auto_fix(
        self,
        input_type: Optional[BaseModel | Type] = None,
        output_type: Optional[BaseModel | Type] = None,
        notify: bool = True,
        max_retries: int = 1,
        context: str = "",
        run_fixed_code: bool = True,
        guardrail: bool = True,
    ):
        """
        Decorator that:
          1. Intercepts exceptions in the target function.
          2. Calls `_build_incident` to obtain a JSON‐serializable dict plus a Pydantic model.
          3. Uses LLM agents (fixer_agent + guardian_agent) to generate and optionally
             execute a patch.
          4. Updates the Incident model with the fix response and notifies if requested.

        Args:
            input_type: Pydantic model or Python type representing the input schema.
            output_type: Pydantic model or Python type representing the output schema.
            notify: If True, calls `incident_handler.notify_autofix_incident`.
            max_retries: Maximum number of attempts if the guardian rejects a patch.
            context: Additional textual context for the LLM.
            run_fixed_code: If True, executes the generated patch in the local environment.
            guardrail: If True, validates the patch with `guardian_agent` before execution.
        """

        # Serialize Pydantic models (input/output) into JSON strings if provided
        if input_type and isinstance(input_type, BaseModel):
            serialized_input_type = input_type.model_dump_json()
        else:
            serialized_input_type = str(input_type)

        if output_type and isinstance(output_type, BaseModel):
            serialized_output_type = output_type.model_dump_json()
        else:
            serialized_output_type = str(output_type)

        def decorator(func: Callable):
            is_async = asyncio.iscoroutinefunction(func)

            async def async_error_handler(incident_model: Any, *args, **kwargs):
                """
                Receives:
                  - incident_model: the Pydantic Incident instance from `_build_incident`
                  - *args, **kwargs: original call arguments
                Attempts to generate and (optionally) execute a patch in a retry loop.
                """
                try:
                    incident_dict = incident_model.model_dump()
                    module_path = incident_dict["module_path"]
                    fixed_function_output = None
                    execution_error = None
                    fix_applied = False

                    # Combine global and function‐specific context for the LLM
                    current_context = (
                        f"Global context: {self.global_context}\n"
                        f"Function context: {context}"
                    )

                    # Retry loop for generating a valid patch
                    for attempt in range(max_retries):
                        try:
                            # 1) Generate the patch via fixer_agent
                            prompt = INCIDENT_RESPONDER_PROMPT.format(
                                source_code=incident_dict["source_code"],
                                module_path=module_path,
                                docstring=incident_dict["docstring"],
                                error=incident_dict["error_type"],
                                error_message=incident_dict["error"],
                                traceback=incident_dict["traceback"],
                                security_report=incident_dict.get(
                                    "security_report", {}
                                ),
                                args=incident_dict["args"],
                                kwargs=incident_dict["kwargs"],
                                context=current_context,
                                output_type=serialized_output_type,
                            )
                            fixed_code = await self.agent_factory.fixer_agent.run(
                                prompt
                            )
                            fixed_code = extract_formatted_text(fixed_code, "python")
                            # 2) If guardrail is enabled, validate with guardian_agent
                            if guardrail:
                                task_prompt = GUARDIAN_TASK.format(
                                    input_type=serialized_input_type,
                                    output_type=serialized_output_type,
                                    error=incident_dict["traceback"],
                                    buggy_code=incident_dict["source_code"],
                                    corrected_code=fixed_code,
                                )
                                raw_judgment = (
                                    await self.agent_factory.guardian_agent.run(
                                        task_prompt
                                    )
                                )
                                judgment = extract_formatted_text(raw_judgment, "json")
                                if not judgment.get("verdict", False):
                                    # Guardian rejected the patch:
                                    if attempt == max_retries - 1:
                                        raise RuntimeError(
                                            "Guardian rejected the patch."
                                        )
                                    continue  # try again

                            # 3) If requested, execute the patch in the local environment
                            if run_fixed_code:
                                (
                                    fixed_function_output,
                                    execution_error,
                                ) = await self._execute_fix_in_current_env(
                                    fixed_code, func, args, kwargs
                                )
                                fix_applied = execution_error is None
                                if fix_applied:
                                    self.__display_applied_fix(
                                        fixed_code, incident_dict["id"]
                                    )

                                # Optionally generate and write a diff file
                                try:
                                    with open(module_path) as f_mod:
                                        module_src = f_mod.read()
                                    patched_full = module_src.replace(
                                        incident_dict["source_code"].strip(),
                                        fixed_code.strip(),
                                    )
                                    base_no_ext = module_path.rsplit(".py", 1)[0]
                                    out_path = (
                                        f"{base_no_ext}_fix_{incident_dict['id']}.py"
                                    )
                                    with open(out_path, "w") as f_out:
                                        f_out.write(
                                            black.format_str(
                                                patched_full, mode=black.FileMode()
                                            )
                                        )
                                except Exception:
                                    # If diff writing fails, continue anyway
                                    pass

                            # 4) Update the Pydantic Incident with the patch response
                            updated_incident = (
                                self.incident_handler.update_incident_with_response(
                                    incident_model,
                                    fixed_code,
                                    fixed_function_output,
                                    str(execution_error) if execution_error else None,
                                    fix_applied,
                                )
                            )
                            logger.warning(
                                f"Incident response: {updated_incident.model_dump()}"
                            )
                            break  # exit retry loop on success

                        except Exception as patch_err:
                            logger.error(
                                f"Fix attempt {attempt+1}/{max_retries} raised: {patch_err}"
                            )
                            if attempt == max_retries - 1:
                                raise

                    # 5) If requested, notify about the auto‐fix incident
                    if notify:
                        self.incident_handler.notify_autofix_incident(updated_incident)

                    return fixed_function_output

                except Exception as e:
                    logger.error(f"Performing the autofix raised error: {e}")
                    raise IncidentException(f"Performing the autofix raised error: {e}")

            def sync_error_handler(incident_model: Any, *args, **kwargs):
                """
                For synchronous functions, invokes `async_error_handler` on an event loop.
                """
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(
                                    async_error_handler(incident_model, *args, **kwargs)
                                )
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            return future.result()
                    else:
                        return asyncio.run(
                            async_error_handler(incident_model, *args, **kwargs)
                        )
                except RuntimeError:
                    return asyncio.run(
                        async_error_handler(incident_model, *args, **kwargs)
                    )

            if is_async:

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        # Build incident_dict and Pydantic model
                        incident_model = self._build_incident(
                            func,
                            exc,
                            args,
                            kwargs,
                            serialized_input_type,
                            serialized_output_type,
                        )
                        # Await the async_error_handler
                        return await async_error_handler(
                            incident_model, *args, **kwargs
                        )

                return async_wrapper

            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except Exception as exc:
                        # Build incident_dict and Pydantic model
                        incident_model = self._build_incident(
                            func,
                            exc,
                            args,
                            kwargs,
                            serialized_input_type,
                            serialized_output_type,
                        )
                        # Call sync_error_handler (which in turn awaits async_error_handler)
                        return sync_error_handler(incident_model, *args, **kwargs)

                return sync_wrapper

        return decorator

    def _build_incident(
        self,
        func: Callable,
        exc: Exception,
        args: Tuple[Any, ...],
        kwargs: dict,
        serialized_input_type: str,
        serialized_output_type: str,
    ) -> Tuple[dict, Any]:
        tb_str = traceback.format_exc()

        module = func.__module__
        module = importlib.import_module(module)
        module_path = module.__file__
        # Create a Pydantic Incident model for update_incident_with_response
        incident_model = BaseIncidentHandler().make_incident(
            func,
            module_path,
            args,
            kwargs,
            "auto_fix",
            exc,
            serialized_input_type,
            serialized_output_type,
            tb_str,
        )

        return incident_model

    @staticmethod
    def __display_applied_fix(fixed_code: str, incident_id: str = "") -> None:
        """
        Prints the applied fix in a formatted manner to the console.

        Args:
            fixed_code (str): The corrected code to display.
            incident_id (str, optional): Identifier for the incident. Defaults to "".
        """
        header = f"\n{'='*60}\n✅ Applied Fix"
        if incident_id:
            header += f" for Incident ID: {incident_id}"
        header += f"\n{'='*60}"
        print(header)
        print(f"\n{fixed_code}\n")
        print(f"{'='*60}\n")

    @staticmethod
    async def _execute_fix_in_current_env(
        fixed_code: str, original_func: Callable, args: Tuple[Any, ...], kwargs: dict
    ) -> Tuple[Any | None, None | Exception]:
        """
        Executes the patched code inline in the namespace of the original module.
        Returns (result, exception). If exception is None, the patch ran successfully.

        Args:
            fixed_code: The source code of the patched function.
            original_func: The original function object being replaced.
            args: Positional arguments for the patched function call.
            kwargs: Keyword arguments for the patched function call.

        Returns:
            (result, exception) – if exception is None, `result` is the return value.
        """
        module_obj = importlib.import_module(original_func.__module__)
        namespace = {
            name: getattr(module_obj, name)
            for name in dir(module_obj)
            if not name.startswith("__")
            and not inspect.isbuiltin(getattr(module_obj, name))
        }
        try:
            exec(fixed_code, namespace)
            patched_fn = namespace.get(original_func.__name__)
            if not callable(patched_fn):
                raise RuntimeError("Error: patched code is not callable")
            if asyncio.iscoroutinefunction(patched_fn):
                result = await patched_fn(*args, **kwargs)
            else:
                result = patched_fn(*args, **kwargs)
            return result, None
        except Exception as ex:
            return None, ex

"""
This module provides a basic incident handler implementation for managing
runtime incidents in a project. Subclass this for more advanced integrations
(e.g., persisting to a database, custom notification systems, or additional behaviors).
"""

import logging
import uuid
from datetime import datetime, timezone
import inspect
from typing import Any, Optional

from local_runtime_autofixer.utils.models import Incident, IncidentResponse


def _default_formatter() -> str:
    """
    Returns the default timestamp format in ISO 8601 with UTC.
    """
    return datetime.now(timezone.utc).isoformat() + "Z"


class BaseIncidentHandler:
    """
    Base class for handling incidents. Creates, updates, and notifies about incidents.

    This implementation:
      - Generates a unique incident ID using a UUID prefix and ISO timestamp.
      - Populates an Incident model with function details, args, kwargs, and traceback.
      - Allows customization of logging handlers and verbosity.
      - Provides methods to notify about both auto-fix and security incidents.
      - Meant to be subclassed for persistence, custom notifications, or other behaviors.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        log_file_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the BaseIncidentHandler.

        Args:
            logger (Optional[logging.Logger]): A preconfigured logger instance.
                If None, a default logger named "incident_handler" will be created.
            log_file_path (Optional[str]): If provided, a FileHandler is attached
                to write logs to this path in addition to console output.
            verbose (bool): If True, incident notifications will also be printed
                to stdout. Otherwise, only logged via the logger.
        """
        if logger is None:
            logger = logging.getLogger("incident_handler")
            logger.setLevel(logging.INFO)

            # Avoid propagation to root logger
            logger.propagate = False

            if not logger.handlers:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

                if log_file_path:
                    file_handler = logging.FileHandler(log_file_path)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)

        self.logger = logger
        self.verbose = verbose

    def make_incident(
        self,
        func: Any,
        module_path: str,
        args: tuple,
        kwargs: dict,
        incident_type: str,
        exception: Optional[Exception] = None,
        serialized_input_type: str = "",
        serialized_output_type: str = "",
        actual_traceback: Optional[str] = None,
    ) -> Incident:
        """
        Create a new Incident object given a caught exception or error event.

        Args:
            func (Any): The function object where the incident occurred.
            args (tuple): Positional arguments that were passed to the function.
            kwargs (dict): Keyword arguments that were passed to the function.
            incident_type (str): A string describing the type of incident.
            exception (Optional[Exception]): The Exception object that was raised (if any).
            serialized_input_type (str): String representation of the function's input schema or type.
            serialized_output_type (str): String representation of the function's output schema or type.
            origin_id (Optional[str]): If this incident was triggered by another incident, its ID.
            security_report (Optional[Dict[str, Any]]): A dictionary describing any security findings.
            actual_traceback (Optional[str]): The full traceback string of the error.

        Returns:
            Incident: An Incident model instance populated with all available information.
        """
        if func is None:
            raise ValueError(
                "Function reference cannot be None when creating an incident."
            )

        unique_id = f"incident_{uuid.uuid4().hex[:8]}_{_default_formatter()}"
        timestamp = _default_formatter()

        try:
            source_code = inspect.getsource(func)
        except (OSError, TypeError):
            source_code = ""

        incident_data = {
            "id": unique_id,
            "type": incident_type,
            "timestamp": timestamp,
            "function_name": getattr(func, "__name__", str(func)),
            "error": str(exception) if exception else None,
            "error_type": type(exception).__name__ if exception else None,
            "serialized_input_type": serialized_input_type,
            "serialized_output_type": serialized_output_type,
            "traceback": actual_traceback,
            "args": [repr(a) for a in args],
            "kwargs": {k: repr(v) for k, v in kwargs.items()},
            "source_code": source_code,
            "docstring": inspect.getdoc(func) or "",
            "incident_response": None,
            "module_path": module_path,
        }

        incident = Incident(**incident_data)
        return incident

    def update_incident_with_response(
        self,
        incident: Incident,
        fix_solution: str,
        fix_output: Optional[str] = None,
        fix_error: Optional[str] = None,
        fix_applied: Optional[bool] = None,
    ) -> Incident:
        """
        Update an existing Incident with the results of an attempted fix.

        Args:
            incident (Incident): The original Incident object to update.
            fix_solution (str): The source code of the proposed fix.
            fix_output (Optional[str]): The output produced by running the patched code (if any).
            fix_error (Optional[str]): Any error encountered when running the patched code (if any).
            fix_applied (Optional[bool]): Boolean indicating whether the fix was applied.

        Returns:
            Incident: The updated Incident instance with a nested IncidentResponse.

        Raises:
            ValueError: If the provided incident has no valid ID.
        """
        if not getattr(incident, "id", None):
            raise ValueError("Cannot update an incident without a valid ID.")

        response_id = f"incident_response_{uuid.uuid4().hex[:8]}_{_default_formatter()}"
        response_timestamp = _default_formatter()

        incident_response = IncidentResponse(
            id=response_id,
            timestamp=response_timestamp,
            incident_id=incident.id,
            fix_solution=fix_solution,
            fix_output=fix_output,
            fix_error=fix_error,
            fix_applied=fix_applied,
        )

        incident.incident_response = incident_response
        return incident

    def notify_autofix_incident(self, incident: Incident):
        """
        Notify about an auto-fix incident by logging detailed information.

        Subclasses can override this method to integrate with email, Slack, or other
        notification channels.

        Args:
            incident (Incident): The Incident object updated with a fix response.
        """
        if not incident:
            return

        incident_data = incident.model_dump()
        response = incident_data.get("incident_response", {})

        # Log summary
        self.logger.warning(f"[AutoFix] Incident ID: {incident_data['id']}")
        self.logger.warning(f"Function: {incident_data['function_name']}")
        self.logger.warning(f"Error: {incident_data['error']}")
        self.logger.warning(f"Fix applied: {response.get('fix_applied', False)}")

        # Optionally print if verbose
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"[AutoFix Incident] ID: {incident_data['id']}")
            print(f"Function: {incident_data['function_name']}")
            print(f"Error: {incident_data['error']}")
            print(f"Fix applied: {response.get('fix_applied', False)}")

            if response.get("fix_solution"):
                print(f"\nFix solution code:\n{response['fix_solution']}")
            if response.get("fix_output"):
                print(f"\nFix output:\n{response['fix_output']}")
            if response.get("fix_error"):
                print(f"\nFix error:\n{response['fix_error']}")
            print("=" * 60 + "\n")

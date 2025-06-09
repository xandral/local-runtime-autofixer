import logging

import pytest

from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler
from local_runtime_autofixer.utils.models import Incident, IncidentResponse


def sample_function(x, y):
    """Adds two numbers"""
    return x + y


def test_make_incident_creates_valid_incident():
    handler = BaseIncidentHandler(verbose=False)
    args = (1, 2)
    kwargs = {"z": 3}
    incident = handler.make_incident(
        func=sample_function,
        module_path="test.module",
        args=args,
        kwargs=kwargs,
        incident_type="TestError",
        exception=ValueError("Something went wrong"),
        serialized_input_type="int, int",
        serialized_output_type="int",
        actual_traceback="Traceback...",
    )

    assert isinstance(incident, Incident)
    assert incident.id.startswith("incident_")
    assert incident.function_name == "sample_function"
    assert incident.error == "Something went wrong"
    assert incident.traceback == "Traceback..."
    assert incident.module_path == "test.module"
    assert incident.serialized_input_type == "int, int"


def test_update_incident_with_response_populates_response():
    handler = BaseIncidentHandler()
    incident = handler.make_incident(
        func=sample_function,
        module_path="test.module",
        args=(1,),
        kwargs={},
        incident_type="FixTest",
        exception=None,
    )

    updated = handler.update_incident_with_response(
        incident=incident,
        fix_solution="return x + y",
        fix_output="3",
        fix_error=None,
        fix_applied=True,
    )

    assert isinstance(updated.incident_response, IncidentResponse)
    assert updated.incident_response.fix_applied is True
    assert updated.incident_response.fix_solution == "return x + y"


def test_notify_autofix_incident_logs_warning(caplog):
    logger = logging.getLogger("incident_handler")
    logger.setLevel(logging.WARNING)

    # Clear existing handlers and add caplog.handler
    logger.handlers.clear()
    logger.addHandler(caplog.handler)
    logger.propagate = False  # Avoid duplicate logs

    handler = BaseIncidentHandler(logger=logger, verbose=False)

    incident = handler.make_incident(
        func=sample_function,
        module_path="test.module",
        args=(1,),
        kwargs={},
        incident_type="FixTest",
    )

    handler.update_incident_with_response(
        incident=incident,
        fix_solution="return x + y",
        fix_output="3",
        fix_error=None,
        fix_applied=True,
    )

    with caplog.at_level(logging.WARNING, logger="incident_handler"):
        handler.notify_autofix_incident(incident)

    assert "[AutoFix] Incident ID" in caplog.text
    assert "Function: sample_function" in caplog.text
    assert "Fix applied: True" in caplog.text


def test_make_incident_raises_if_func_is_none():
    handler = BaseIncidentHandler()
    with pytest.raises(ValueError):
        handler.make_incident(
            func=None,
            module_path="",
            args=(),
            kwargs={},
            incident_type="Invalid",
        )

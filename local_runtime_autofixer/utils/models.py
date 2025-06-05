from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class FunctionDetail(BaseModel):
    """
    Represents the details of a function, including its source code
    and optional docstring.
    """

    source_code: str = Field(
        ..., description="The full source code of the function as a string."
    )
    docstring: Optional[str] = Field(
        None, description="The function's docstring, if one exists."
    )


class IncidentResponse(BaseModel):
    """
    Captures the result of an attempted automatic fix for an incident.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this IncidentResponse (e.g., 'incident_response_20250605123045').",
    )
    incident_id: str = Field(
        ...,
        description="The ID of the related Incident (e.g., 'incident_20250605122930').",
    )
    timestamp: datetime | str = Field(
        ..., description="UTC timestamp (ISO 8601) when this response was generated."
    )
    fix_solution: Optional[str] = Field(
        None,
        description=(
            "The source code string containing the proposed fix. "
            "This may be None if no fix was generated."
        ),
    )
    fix_output: Optional[Any] = Field(
        None,
        description=(
            "The output (return value) produced by running the patched code, "
            "if execution was attempted."
        ),
    )
    fix_applied: bool = Field(
        ...,
        description=(
            "Boolean flag indicating whether the fix was successfully applied "
            "to the original code."
        ),
    )


class Incident(BaseModel):
    """
    Represents a runtime incident (error/exception) that occurred during
    function execution, along with contextual data and an optional response
    (e.g., an automatic fix).
    """

    type: str = Field(
        ...,
        description=(
            "A short string describing the category or type of incident "
            "(e.g., 'auto_fix', 'security_check')."
        ),
    )
    id: str = Field(
        ...,
        description="Unique identifier for this Incident (e.g., 'incident_20250605122930').",
    )
    timestamp: datetime | str = Field(
        ..., description="UTC timestamp (ISO 8601) when the incident was created."
    )
    function_name: str = Field(
        ..., description="Name of the function where the incident occurred."
    )
    module_path: str = Field(
        ..., description="Path to the module that contains the function"
    )
    error: Optional[str] = Field(
        None, description="The string representation of the exception or error message."
    )
    error_type: Optional[str] = Field(
        None,
        description="The class name of the exception or error that was raised (e.g., 'ZeroDivisionError').",
    )
    serialized_input_type: Optional[str] = Field(
        None,
        description="String serialization of the function's input schema or expected input type.",
    )
    serialized_output_type: Optional[str] = Field(
        None,
        description="String serialization of the function's output schema or expected output type.",
    )
    traceback: Optional[str] = Field(
        None, description="Full traceback of the exception as a string."
    )
    args: List[str] = Field(
        ...,
        description=(
            "List of repr() strings for each positional argument passed to the function "
            "when the incident occurred."
        ),
    )
    kwargs: Dict[str, str] = Field(
        ...,
        description=(
            "Dictionary mapping keyword argument names to repr() strings for each "
            "keyword argument passed to the function."
        ),
    )
    source_code: Optional[str] = Field(
        "",
        description="The source code of the function in which the incident occurred.",
    )
    docstring: Optional[str] = Field(
        "", description="The docstring of the function (if present)."
    )
    incident_response: Optional[IncidentResponse] = Field(
        None,
        description=(
            "Nested IncidentResponse object holding details of any automated fix attempt. "
            "Will be None until an update with a fix occurs."
        ),
    )

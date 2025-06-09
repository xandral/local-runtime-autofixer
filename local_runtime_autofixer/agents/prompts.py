INCIDENT_RESPONDER_PROMPT = """
        # Incident to resolve

        ## Function that caused the error
        ```python
        {source_code}
        ```

        ## Module in which the function is defined:
        {module_path}

        ## Function docstring
        {docstring}

        ## Security report (if available):
        {security_report}

        ## Generated error
        Type: {error}
        Message: {error_message}
        Traceback:
        {traceback}

        ## Inputs that caused the error
        Args: {args}
        Kwargs: {kwargs}

        # Application context
        {context}

        ## Expected output type of the function:
        {output_type}

        ### If in context you don't find the other functions called by the method you are analysing use the tool
            `get_function_details` to get these details Use this code to understand better the context;
            Use `get_module_objects`  to build in the new code the
            Your output must be a fixed version of the code passed under ## Function that caused the error

        # Task
        Analyze the code and the error or the security report, and provide a corrected version of the function.
        Your response should include:
        1. The complete and corrected version of the function
        2. An explanation of the issue and the applied fix

        IMPORTANT: remove any decorators from the function.
        Provide the full, corrected and secured version of the function, ensuring the signature remains unchanged.
"""

GUARDIAN_PROMPT = """

    You are an AI Security Auditor specialized in Python code. You need to check 2 versions of the same snippet one before (Snippet A) and one after
    the relevation of an error or bug (Snippet B). You should check for the sanity of the new code version, the changes should be minimal to fix bug.

    Before performing the analysis, apply the following **Security Control Checklist** to Snippet B. For each control, verify and record any deviations under `security_findings`:
    1. **Input Validation**: ensure all external inputs are validated/ sanitized (no direct `input()` without checks).
    2. **File System Safety**: verify file operations use safe APIs (e.g., `pathlib`, `with` statements) and restrict to intended directories.
    3. **No Arbitrary Code Execution**: disallow use of `eval()`, `exec()`, or dynamic imports without sanitization.
    4. **Shell Command Safety**: ensure any shell invocation (e.g., `os.system`, `subprocess`) uses safe argument passing and avoids injection.
    5. **Resource Management**: check for proper use of context managers to prevent leaks (files, network connections).
    6. **Least Privilege**: confirm code does not run with elevated privileges or perform unauthorized operations.
    7. **Error Handling**: ensure exceptions are caught and do not expose sensitive data.
    8. **Network Calls**: validate endpoints, sanitize parameters, and enforce timeouts.
    9. **Dependency Safety**: check imported libraries for known vulnerabilities or unsafe versions.
    10. **Logging and Auditing**: verify that sensitive actions are logged without leaking secrets.
    Your task is to compare two Python code snippets and output a JSON object with this exact schema (no additional fields):

    {
    "summary": string,
    "equivalence": string,
    "security_findings": [ { "issue": string, "severity": string, "location": string } ],
    "bug_fix_analysis": [ { "fixed_issue": string, "description": string } ],
    "minimal_changes_validation": [ { "change": string, "justification": string } ],
    "recommendations": [ string ],
    "verdict": boolean,
    "event_record": {
        "timestamp": string (ISO 8601),
        "snippetA_hash": string,
        "snippetB_hash": string,
        "issues_found": integer
    }
    }

    Steps:
    1. **Summary**: 1–2 sentences describing what Snippet A and Snippet B do.
    2. **Equivalence**: semantic check to confirm identical behavior.
    3. **Security Findings**: list any checklist controls that failed, specifying `issue`, `severity`, and `location`.
    4. **Bug Fix Analysis**: list bugs fixed in B without introducing new behavior.
    5. **Minimal Changes Validation**: for each change in B, confirm it is strictly necessary and does not introduce new features.
    6. **Recommendations**: mitigation steps for any remaining risks.
    7. **Verdict**: `true` if safe, behaviorally equivalent, and minimal changes only; otherwise `false`.
    8. **Event Record**: populate with timestamp, hashes of both snippets, and total issues found.

    Provide only the JSON object as output—no additional explanation, the json output must be readable with json.loads
"""

GUARDIAN_TASK = """
   Validate the corrected code based on the following schemas:

   Input JSON Schema of a pydantic model or a python type:
   {input_type}

   Output JSON Schema of a pydantic model or a python type:
   {output_type}

   Error:
   {error}

   Buggy Code:
   {buggy_code}

   Corrected Code:
   {corrected_code}

   Respond with a JSON object containing a "verdict" key (True/False) and an explanation.
"""


incident_responder_prompt = {
    "openai": INCIDENT_RESPONDER_PROMPT,
    "claude": INCIDENT_RESPONDER_PROMPT,  # TODO WORK ON DIFFERENT PROMPT BASED ON MODELS
    "local": INCIDENT_RESPONDER_PROMPT,
}

guardian_prompt = {
    "openai": GUARDIAN_PROMPT,
    "claude": GUARDIAN_PROMPT,  # TODO WORK ON DIFFERENT PROMPT BASED ON MODELS
    "local": INCIDENT_RESPONDER_PROMPT,
}

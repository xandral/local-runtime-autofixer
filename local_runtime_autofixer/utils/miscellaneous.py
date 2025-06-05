import json
import re


def extract_formatted_text(text: str, text_type: str) -> str:
    pattern = rf"```{text_type}\s*(.*?)\s*```"

    if text_type == "json":
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return json.loads(match.group(1))
        else:
            return json.loads(text)

    elif text_type == "python":
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else text

    else:
        raise NotImplementedError(f"Text type {text_type} not implemented")

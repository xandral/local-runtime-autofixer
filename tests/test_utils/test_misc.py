import pytest
import json
from local_runtime_autofixer.utils.miscellaneous import extract_formatted_text


def test_extract_json_inside_code_block():
    text = """Some intro
```json
{"key": "value", "num": 42}
```"""
    result = extract_formatted_text(text, "json")
    assert result == {"key": "value", "num": 42}


def test_extract_raw_json_when_no_code_block():
    text = '{"key": "value", "num": 42}'
    result = extract_formatted_text(text, "json")
    assert result == {"key": "value", "num": 42}


def test_extract_python_code_from_code_block():
    text = """```python
        def hello():
            return "world"
        ```"""
    result = extract_formatted_text(text, "python")
    assert "def hello()" in result
    assert 'return "world"' in result


def test_return_original_text_if_no_python_code_block():
    text = "print('Hello, world!')"
    result = extract_formatted_text(text, "python")
    assert result == text


def test_raises_on_unsupported_text_type():
    with pytest.raises(NotImplementedError) as excinfo:
        extract_formatted_text("some text", "yaml")
    assert "Text type yaml not implemented" in str(excinfo.value)


def test_raises_on_invalid_json():
    broken_json = """```json
        {invalid json}
        ```"""
    with pytest.raises(json.JSONDecodeError):
        extract_formatted_text(broken_json, "json")

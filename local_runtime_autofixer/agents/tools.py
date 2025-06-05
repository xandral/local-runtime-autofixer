import inspect
import importlib
from local_runtime_autofixer.utils.models import FunctionDetail
from langchain_core.tools import tool


@tool
async def get_function_details(module: str, function_name: str) -> FunctionDetail:
    """This tools takes in input a name of a python function in the current module and gets it's source code and if exists the docstring"""
    module = importlib.import_module(module)
    f = getattr(module, function_name)
    code = inspect.getsource(f)
    doc = inspect.getdoc(f)
    return FunctionDetail(source_code=code, docstring=doc)

import asyncio

from pydantic import BaseModel

from local_runtime_autofixer.autofixer import LocalIncidentResponder
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler

incindent_handler = BaseIncidentHandler()

responder = LocalIncidentResponder(incindent_handler)


class InputModel(BaseModel):
    a: int
    b: int


class OutputModel(BaseModel):
    result: int


@responder.auto_fix(input_type=InputModel, output_type=int, notify=False, max_retries=1)
async def faulty(a, b):
    # 'c' is undefined, triggers NameError
    return a + c


@responder.auto_fix(input_type=InputModel, output_type=int, notify=False, max_retries=1)
async def faulty2(a, b):
    x = a + c
    return await faulty(a, c)


asyncio.run(faulty2(1, 2))

from pydantic import BaseModel

from local_runtime_autofixer.autofixer import LocalIncidentResponder
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler

responder = LocalIncidentResponder()


class InputModel(BaseModel):
    a: int
    b: int


class OutputModel(BaseModel):
    result: int


@responder.auto_fix(input_type=InputModel, output_type=int, notify=True, max_retries=1)
def faulty(a, b):
    # 'c' is undefined, triggers NameError
    return a + c


# _, r = asyncio.run(faulty(1, 2))
# print("RESULT AFTER FIX", r)
faulty(2, 3)

from local_runtime_autofixer.autofixer import LocalIncidentResponder
from local_runtime_autofixer.utils.incident_handler import BaseIncidentHandler
import asyncio
from pydantic import BaseModel

incindent_handler = BaseIncidentHandler()

responder = LocalIncidentResponder(incindent_handler)


class InputModel(BaseModel):
    a: int
    b: int


@responder.auto_fix(
    input_type=InputModel, output_type=int | None, notify=False, max_retries=1
)
def division(a: int, b: int):
    """Division function if b=0 returns None"""
    result = a / b
    print("RESULT:", result)
    return result


async def main():
    # Use asyncio.gather to run both tasks concurrently
    # This way if one fails, the other can still run
    results = await asyncio.gather(
        division(1, 0),
        division(4, 2),
        return_exceptions=True,  # This prevents exceptions from being raised
    )


if __name__ == "__main__":
    asyncio.run(main())

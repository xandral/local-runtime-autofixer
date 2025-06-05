from example_autofix import faulty
import asyncio


def call1():
    return faulty(2, 3)


def call2():
    return call1()


x = call2()

pass

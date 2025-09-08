import asyncio
from collections import defaultdict
from typing import Callable


class TaskManager:
    def __init__(self):
        self._tasks = defaultdict(set)

    def create_task(
        self, fn: Callable, interval: int | None = None, exclusive: bool = False, *args
    ):
        if exclusive and fn.__name__ in self._tasks:
            for t in self._tasks[fn.__name__]:
                t.cancel()
            self._tasks[fn.__name__].clear()

        if interval:
            task = asyncio.create_task(
                _set_interval(fn, interval, *args), name=fn.__name__
            )
        else:
            task = asyncio.create_task(fn(*args), name=fn.__name__)
        task.add_done_callback(lambda t: self._remove_task(fn.__name__, t))
        self._tasks[fn.__name__].add(task)

    def cancel_all(self):
        for task_list in self._tasks.values():
            for t in task_list:
                t.cancel()
        self._tasks.clear()

    def cancel(self, fn_name: str):
        for task in self._tasks[fn_name]:
            task.cancel()
        self._tasks[fn_name].clear()

    def _remove_task(self, task_name: str, task: asyncio.Task):
        task_set = self._tasks[task_name]
        if task_set:
            task_set.remove(task)


async def _set_interval(fn: Callable, interval: int, *args):
    while True:
        await fn(*args)
        await asyncio.sleep(interval)

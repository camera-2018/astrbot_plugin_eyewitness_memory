"""Process-local FastBrain scheduling shared by independently loaded plugins.

Vendor this module unchanged in each client. The versioned registry survives
plugin reloads; it stores no prompts or credentials. Other processes/endpoints
are outside its scope. Ordinary requests remain FIFO, with a bounded urgent burst.
"""

from __future__ import annotations

import asyncio
import sys
import types
import weakref
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

_REGISTRY = "_astrbot_fastbrain_priority_queue_v1"
MAX_URGENT_BURST = 3


@dataclass(frozen=True)
class Ticket:
    wait_ms: int
    priority: str


class _Lane:
    def __init__(self):
        self.urgent = deque()
        self.normal = deque()
        self.active = None
        self.burst = 0

    def dispatch(self):
        if self.active is not None:
            return
        for queue in (self.urgent, self.normal):
            while queue and queue[0].cancelled():
                queue.popleft()
        if self.urgent and (not self.normal or self.burst < MAX_URGENT_BURST):
            queue = self.urgent
            self.burst = min(MAX_URGENT_BURST, self.burst + 1)
        elif self.normal:
            queue = self.normal
            self.burst = 0
        else:
            self.burst = 0
            return
        self.active = queue.popleft()
        self.active.set_result(None)

    async def acquire(self, urgent):
        waiter = asyncio.get_running_loop().create_future()
        queue = self.urgent if urgent else self.normal
        queue.append(waiter)
        self.dispatch()
        try:
            await waiter
        except BaseException:
            # Cancellation can race with a grant. Release only our own lease.
            if self.active is waiter:
                self.active = None
            elif waiter in queue:
                queue.remove(waiter)
            self.dispatch()
            raise
        return waiter

    def release(self, waiter):
        if self.active is not waiter:
            raise RuntimeError("FastBrain queue lease mismatch")
        self.active = None
        self.dispatch()


def _lane(endpoint, model):
    registry = sys.modules.get(_REGISTRY)
    if registry is None:
        registry = types.ModuleType(_REGISTRY)
        registry.loops = weakref.WeakKeyDictionary()
        sys.modules[_REGISTRY] = registry
    loop = asyncio.get_running_loop()
    lanes = registry.loops.setdefault(loop, {})
    url = urlsplit(endpoint)
    key = (
        url.scheme.lower(),
        (url.hostname or "").lower(),
        url.port or (443 if url.scheme.lower() == "https" else 80),
        url.path.rstrip("/"),
        model,
    )
    return lanes.setdefault(key, _Lane())


def is_fastbrain(model):
    return "fastbrain" in str(model).casefold()


@asynccontextmanager
async def priority_slot(endpoint, model, *, urgent=False, enabled=True):
    if not enabled or not is_fastbrain(model):
        yield Ticket(0, "unmanaged")
        return
    lane = _lane(endpoint, model)
    loop = asyncio.get_running_loop()
    started = loop.time()
    waiter = await lane.acquire(urgent)
    try:
        yield Ticket(round((loop.time() - started) * 1000), "urgent" if urgent else "normal")
    finally:
        lane.release(waiter)

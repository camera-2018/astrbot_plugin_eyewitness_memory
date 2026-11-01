import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from eyewitness.fastbrain_queue import _lane, priority_slot

URL = "http://queue.test:3067/fastbrain/v1/systemone"
MODEL = "FastBrain-9B"


async def queued_call(
    name, order, *, urgent=False, entered=None, release=None, factory=priority_slot
):
    async with factory(URL, MODEL, urgent=urgent):
        order.append(name)
        if entered:
            entered.set()
        if release:
            await release.wait()


async def test_urgent_overtakes_pending_but_not_running():
    order = []
    entered, release = asyncio.Event(), asyncio.Event()
    active = asyncio.create_task(queued_call("active", order, entered=entered, release=release))
    await entered.wait()
    normal1 = asyncio.create_task(queued_call("n1", order))
    normal2 = asyncio.create_task(queued_call("n2", order))
    urgent = asyncio.create_task(queued_call("gate", order, urgent=True))
    await asyncio.sleep(0)
    assert order == ["active"]
    release.set()
    await asyncio.gather(active, normal1, normal2, urgent)
    assert order == ["active", "gate", "n1", "n2"]


async def test_fairness_serves_normal_after_three_urgent_requests():
    order = []
    async with priority_slot(URL, MODEL):
        normal = asyncio.create_task(queued_call("normal", order))
        urgent = [asyncio.create_task(queued_call(str(i), order, urgent=True)) for i in range(7)]
        await asyncio.sleep(0)
    await asyncio.gather(normal, *urgent)
    assert order == ["0", "1", "2", "normal", "3", "4", "5", "6"]


async def test_pending_timeout_does_not_run_or_release_active_request():
    order = []

    async def expiring():
        async with asyncio.timeout(0.01):
            await queued_call("expired", order, urgent=True)

    async with priority_slot(URL, MODEL):
        task = asyncio.create_task(expiring())
        with pytest.raises(TimeoutError):
            await task
        assert _lane(URL, MODEL).active is not None
        assert not _lane(URL, MODEL).urgent
    await queued_call("next", order)
    assert order == ["next"]


async def test_cancel_racing_with_grant_does_not_leak_lease():
    lane = _lane(URL, MODEL)
    held = await lane.acquire(False)
    waiter = asyncio.create_task(queued_call("cancelled", [], urgent=True))
    await asyncio.sleep(0)
    lane.release(held)  # grant scheduled, waiter has not resumed yet
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    async with asyncio.timeout(0.1):
        async with priority_slot(URL, MODEL):
            pass


async def test_active_cancellation_and_errors_release_only_owned_lease():
    order = []
    entered, release = asyncio.Event(), asyncio.Event()
    active = asyncio.create_task(queued_call("active", order, entered=entered, release=release))
    await entered.wait()
    next_call = asyncio.create_task(queued_call("next", order))
    await asyncio.sleep(0)
    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    await next_call
    with pytest.raises(ValueError):
        async with priority_slot(URL, MODEL):
            raise ValueError("simulated transport failure")
    async with priority_slot(URL, MODEL):
        pass
    assert order == ["active", "next"]


async def test_independent_module_load_and_reload_share_existing_queue():
    # Exactly how two plugin packages load the vendored module under different names.
    path = Path(__file__).parents[1] / "eyewitness/fastbrain_queue.py"
    name = "_test_independent_fastbrain_queue"
    order = []
    async with priority_slot(URL, MODEL):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            assert module._lane(URL, MODEL) is _lane(URL, MODEL)
            pending = asyncio.create_task(
                queued_call("other_plugin", order, factory=module.priority_slot)
            )
            await asyncio.sleep(0)
            assert not order
        finally:
            sys.modules.pop(name, None)
    await pending
    assert order == ["other_plugin"]


async def test_endpoint_isolation_and_canonical_default_port():
    async with priority_slot(URL, MODEL):
        async with priority_slot("http://other.test/v1/systemone", MODEL):
            pass
    assert _lane("http://QUEUE.test:80/v1/", MODEL) is _lane("http://queue.test/v1", MODEL)


async def test_non_fastbrain_provider_is_not_serialized():
    async with priority_slot(URL, "jev-1.0") as ticket:
        async with priority_slot(URL, "jev-1.0"):
            assert ticket.priority == "unmanaged"

from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except Exception:  # pragma: no cover
    # Minimal shims if tenacity is unavailable; avoids import failure
    def retry(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

    def stop_after_attempt(n):  # type: ignore
        return n

    def wait_random_exponential(multiplier=1, max=8):  # type: ignore
        return (multiplier, max)


@contextmanager
def timed(to: dict, key: str) -> Iterator[None]:
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        to[key] = round(dt, 4)

network_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=8),
)

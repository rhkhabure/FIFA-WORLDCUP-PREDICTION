"""Bounded retry behavior for read-only external API requests."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def urlopen_with_retry(
    request: urllib.request.Request,
    *,
    timeout: float,
    context: Any = None,
    retries: int = 2,
    backoff_seconds: float = 0.25,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """Open a request with bounded exponential retry for transient failures."""
    retries = max(0, int(retries))
    for attempt in range(retries + 1):
        try:
            kwargs = {"timeout": timeout}
            if context is not None:
                kwargs["context"] = context
            return opener(request, **kwargs)
        except urllib.error.HTTPError as exc:
            retryable = exc.code in TRANSIENT_HTTP_STATUS
            if not retryable or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries:
                raise
        sleeper(backoff_seconds * (2**attempt))

    raise RuntimeError("request retry loop exited unexpectedly")

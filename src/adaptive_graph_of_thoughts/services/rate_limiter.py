from collections import defaultdict
from typing import DefaultDict
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS


class RateLimiter:
    def __init__(self, max_requests: int, per_seconds: int) -> None:
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._log: DefaultDict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        timestamps = [t for t in self._log[key] if now - t < self.per_seconds]
        self._log[key] = timestamps
        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(now)
        self._log[key] = timestamps
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int, per_seconds: int) -> None:
        super().__init__(app)
        self.limiter = RateLimiter(max_requests, per_seconds)

    async def dispatch(self, request: Request, call_next):
        client_id = request.client.host if request.client else "unknown"
        if not self.limiter.is_allowed(client_id):
            return Response("Too Many Requests", status_code=HTTP_429_TOO_MANY_REQUESTS)
        return await call_next(request)

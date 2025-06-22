import asyncio
from typing import Optional

import psutil
from loguru import logger


class ResourceMonitor:
    """Monitor system resources and enforce simple usage limits."""

    def __init__(self, max_memory_mb: int = 1024, max_cpu_percent: float = 80.0) -> None:
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent

    async def check_resources(self) -> bool:
        """Return False if current usage exceeds configured limits."""
        try:
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            if memory_mb > self.max_memory_mb:
                logger.warning(f"Memory usage high: {memory_mb:.1f}MB")
                return False

            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.max_cpu_percent:
                logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
                return False
            return True
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(f"Resource check failed: {exc}")
            return True

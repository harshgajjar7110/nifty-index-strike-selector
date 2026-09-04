"""
Retry decorator for network-bound operations (yfinance, NSE API).
"""

import time
from functools import wraps
from loguru import logger


def retry(max_retries: int = 3, backoff_seconds: float = 2.0, exceptions=(Exception,)):
    """Decorator that retries a function on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    sleep_time = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
            return None  # unreachable, but satisfies type checker
        return wrapper
    return decorator

"""Async utilities for improved scalability.

Provides async wrappers and utilities for concurrent operations.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, partial
from typing import Any, Callable, List, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global thread pool for running sync code in async context
_executor: Optional[ThreadPoolExecutor] = None


def get_executor(max_workers: int = 10) -> ThreadPoolExecutor:
    """Get the global thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


async def run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a synchronous function in a thread pool executor.
    
    Args:
        func: The synchronous function to run.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        The function's return value.
    """
    loop = asyncio.get_event_loop()
    executor = get_executor()
    
    if kwargs:
        func = partial(func, **kwargs)
    
    return await loop.run_in_executor(executor, func, *args)


def async_wrap(func: Callable[..., T]) -> Callable[..., asyncio.coroutine]:
    """
    Decorator to wrap a synchronous function for async execution.
    
    Args:
        func: The synchronous function to wrap.
        
    Returns:
        An async version of the function.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await run_in_executor(func, *args, **kwargs)
    return wrapper


async def gather_with_concurrency(
    n: int,
    *tasks,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run multiple coroutines with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent tasks.
        *tasks: Coroutines to run.
        return_exceptions: If True, exceptions are returned as results.
        
    Returns:
        List of results from all tasks.
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(sem_task(task) for task in tasks),
        return_exceptions=return_exceptions
    )


async def retry_async(
    func: Callable[..., T],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: The async function to retry.
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries (seconds).
        backoff: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch and retry.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        The function's return value.
        
    Raises:
        The last exception if all retries fail.
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await run_in_executor(func, *args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception


class AsyncBatcher:
    """Batch multiple async requests for efficiency.
    
    Useful for batching LLM calls or database operations.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_wait_time: float = 0.1
    ):
        """
        Initialize the async batcher.
        
        Args:
            batch_size: Maximum batch size before processing.
            max_wait_time: Maximum time to wait for batch to fill.
        """
        self._batch_size = batch_size
        self._max_wait_time = max_wait_time
        self._pending: List[tuple] = []
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
    
    async def add(self, item: Any, processor: Callable) -> Any:
        """
        Add an item to the batch and wait for result.
        
        Args:
            item: Item to process.
            processor: Async function to process the batch.
            
        Returns:
            Result for this item.
        """
        future = asyncio.Future()
        
        async with self._lock:
            self._pending.append((item, future))
            
            if len(self._pending) >= self._batch_size:
                await self._process_batch(processor)
        
        # Wait for result
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            return None
    
    async def _process_batch(self, processor: Callable) -> None:
        """Process the current batch."""
        if not self._pending:
            return
        
        batch = self._pending[:]
        self._pending.clear()
        
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]
        
        try:
            results = await processor(items)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


async def timeout_wrapper(
    coro,
    timeout: float,
    default: Any = None
) -> Any:
    """
    Wrap a coroutine with a timeout.
    
    Args:
        coro: The coroutine to wrap.
        timeout: Timeout in seconds.
        default: Value to return on timeout.
        
    Returns:
        The coroutine result or default on timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default

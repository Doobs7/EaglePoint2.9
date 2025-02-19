import logging
import traceback
import functools
import asyncio

logger = logging.getLogger("meta_agent")
logger.setLevel(logging.DEBUG)  # Adjust as needed
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def log_exception(e, context=""):
    error_message = f"Exception: {str(e)}. Context: {context}\n{traceback.format_exc()}"
    logger.error(error_message)
    return error_message

def catch_and_log(context=""):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    log_exception(e, context or f"Error in {func.__name__}")
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log_exception(e, context or f"Error in {func.__name__}")
                    raise
            return sync_wrapper
    return decorator

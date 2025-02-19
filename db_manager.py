import os
import asyncpg
import asyncio
from dotenv import load_dotenv
from error_handler import log_exception, catch_and_log

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")

pool = None

async def ensure_pool():
    global pool
    if pool is None:
        try:
            pool = await asyncpg.create_pool(dsn=POSTGRES_URL, min_size=1, max_size=10)
        except Exception as e:
            log_exception(e, "Error initializing database connection pool")
            raise

@catch_and_log("Getting connection from pool")
async def get_connection():
    await ensure_pool()
    try:
        conn = await pool.acquire()
        return conn
    except Exception as e:
        log_exception(e, "Error getting connection from pool")
        raise

@catch_and_log("Releasing connection back to pool")
async def release_connection(conn):
    try:
        await pool.release(conn)
    except Exception as e:
        log_exception(e, "Error releasing connection back to pool")
        raise

class DBConnection:
    """
    Async context manager for database connections.
    """
    def __init__(self):
        self.conn = None

    async def __aenter__(self):
        self.conn = await get_connection()
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        await release_connection(self.conn)

@catch_and_log("Executing query")
async def execute_query(query: str, *args):
    """
    Convenience function to execute a query and return the result.
    """
    async with DBConnection() as conn:
        try:
            result = await conn.fetch(query, *args)
            return result
        except Exception as e:
            log_exception(e, f"Error executing query: {query}")
            raise

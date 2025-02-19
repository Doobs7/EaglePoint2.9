import json
import asyncio
from datetime import datetime, timedelta, timezone
import logging
from db_manager import get_connection, release_connection
from error_handler import log_exception, catch_and_log
import aiohttp  # For external data ingestion

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@catch_and_log("Initializing Database")
async def init_db():
    """
    Initializes the agent_logs table with enriched metadata, versioning, and memory layer support.
    """
    conn = await get_connection()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id SERIAL PRIMARY KEY,
                agent_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                content TEXT,
                metadata JSONB,
                version INTEGER DEFAULT 1
            );
        """)
        logger.info("Database initialized for agent_logs.")
    except Exception as e:
        log_exception(e, "Error during init_db")
        raise
    finally:
        await release_connection(conn)

@catch_and_log("Initializing Audit Trail")
async def init_audit_trail():
    """
    Initializes an audit table for event updates.
    """
    conn = await get_connection()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs_audit (
                audit_id SERIAL PRIMARY KEY,
                event_id INTEGER,
                agent_id TEXT,
                old_content TEXT,
                old_metadata JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Audit trail initialized for agent_logs.")
    except Exception as e:
        log_exception(e, "Error during init_audit_trail")
        raise
    finally:
        await release_connection(conn)

@catch_and_log("Logging event")
async def log_event(agent_id, event_type, content, metadata=None, relevance_score=None, tags=None, summary=None, memory_type=None, memory_layer="episodic"):
    """
    Logs an event with enriched metadata, including memory type and layer.
    """
    conn = await get_connection()
    try:
        if metadata is None:
            metadata = {}
        # Use timezone-aware UTC time
        metadata.setdefault('logged_at', datetime.now(timezone.utc).isoformat())
        if relevance_score is not None:
            metadata['relevance_score'] = relevance_score
        if tags is not None:
            metadata['tags'] = tags
        if summary is not None:
            metadata['event_summary'] = summary
        if memory_type is not None:
            metadata['memory_type'] = memory_type
        metadata['memory_layer'] = memory_layer
        metadata_json = json.dumps(metadata)
        await conn.execute(
            "INSERT INTO agent_logs (agent_id, event_type, content, metadata) VALUES ($1, $2, $3, $4)",
            agent_id, event_type, content, metadata_json
        )
        logger.info(f"Event logged for agent {agent_id}: {event_type} - {content}")
    except Exception as e:
        log_exception(e, "Error logging event")
    finally:
        await release_connection(conn)

@catch_and_log("Fetching recent events")
async def get_recent_events(agent_id, limit=5):
    """
    Retrieves the most recent events for the agent.
    """
    conn = await get_connection()
    try:
        rows = await conn.fetch("""
            SELECT timestamp, event_type, content, metadata 
            FROM agent_logs 
            WHERE agent_id = $1 
            ORDER BY timestamp DESC 
            LIMIT $2
        """, agent_id, limit)
        events = []
        for row in rows:
            try:
                metadata_obj = json.loads(row.get("metadata") or "{}")
            except Exception:
                metadata_obj = {}
            events.append(f"{row['timestamp']} - {row['event_type']} (Metadata: {metadata_obj}): {row['content']}")
        summary = "\n".join(events)
        logger.info(f"Fetched recent events for agent {agent_id}.")
        return summary
    except Exception as e:
        log_exception(e, "Error fetching recent events")
        return "Error retrieving logs."
    finally:
        await release_connection(conn)

@catch_and_log("Pruning old events")
async def prune_old_events(agent_id, retention_days=30):
    """
    Prunes events older than a certain retention period.
    """
    conn = await get_connection()
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        result = await conn.execute("""
            DELETE FROM agent_logs 
            WHERE agent_id = $1 AND timestamp < $2
        """, agent_id, cutoff_date)
        logger.info(f"Pruned events for agent {agent_id} older than {retention_days} days.")
        return result
    except Exception as e:
        log_exception(e, "Error pruning old events")
    finally:
        await release_connection(conn)

@catch_and_log("Archiving old events")
async def archive_old_events(agent_id, retention_days=30):
    """
    Archives events older than a certain period by moving them to an archive table.
    """
    conn = await get_connection()
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs_archive (
                id SERIAL PRIMARY KEY,
                agent_id TEXT,
                timestamp TIMESTAMP,
                event_type TEXT,
                content TEXT,
                metadata JSONB,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await conn.execute("""
            INSERT INTO agent_logs_archive (agent_id, timestamp, event_type, content, metadata)
            SELECT agent_id, timestamp, event_type, content, metadata
            FROM agent_logs
            WHERE agent_id = $1 AND timestamp < $2;
        """, agent_id, cutoff_date)
        await conn.execute("""
            DELETE FROM agent_logs
            WHERE agent_id = $1 AND timestamp < $2;
        """, agent_id, cutoff_date)
        logger.info(f"Archived events for agent {agent_id} older than {retention_days} days.")
    except Exception as e:
        log_exception(e, "Error archiving old events")
    finally:
        await release_connection(conn)

@catch_and_log("Updating event")
async def update_event(agent_id, event_id, new_content=None, new_metadata=None):
    """
    Updates an event’s content and metadata, increments its version, and logs the change.
    """
    conn = await get_connection()
    try:
        rows = await conn.fetch("SELECT content, metadata, version FROM agent_logs WHERE id = $1 AND agent_id = $2", event_id, agent_id)
        if not rows:
            logger.info(f"No event found with id {event_id} for agent {agent_id}.")
            return
        old_content = rows[0]['content']
        old_metadata = rows[0]['metadata']
        old_version = rows[0]['version']
        new_version = old_version + 1
        if new_metadata:
            new_metadata = json.dumps(new_metadata)
        await conn.execute("""
            UPDATE agent_logs
            SET content = COALESCE($2, content),
                metadata = COALESCE($3, metadata),
                version = $4
            WHERE id = $1 AND agent_id = $5;
        """, event_id, new_content, new_metadata, new_version, agent_id)
        await conn.execute("""
            INSERT INTO agent_logs_audit (event_id, agent_id, old_content, old_metadata)
            VALUES ($1, $2, $3, $4);
        """, event_id, agent_id, old_content, old_metadata)
        logger.info(f"Updated event {event_id} for agent {agent_id}. New version: {new_version}")
    except Exception as e:
        log_exception(e, "Error updating event")
    finally:
        await release_connection(conn)

@catch_and_log("Deleting event")
async def delete_event(agent_id, event_id):
    """
    Deletes an event by its ID.
    """
    conn = await get_connection()
    try:
        await conn.execute("""
            DELETE FROM agent_logs
            WHERE id = $1 AND agent_id = $2;
        """, event_id, agent_id)
        logger.info(f"Deleted event {event_id} for agent {agent_id}.")
    except Exception as e:
        log_exception(e, "Error deleting event")
    finally:
        await release_connection(conn)

@catch_and_log("Natural Language Query for Events")
async def query_events_nl(agent_id, nl_query):
    """
    Provides a natural language interface to query events using a simple keyword search.
    """
    conn = await get_connection()
    try:
        query = """
            SELECT timestamp, event_type, content, metadata 
            FROM agent_logs 
            WHERE agent_id = $1 AND content ILIKE '%' || $2 || '%' 
            ORDER BY timestamp DESC;
        """
        rows = await conn.fetch(query, agent_id, nl_query)
        events = []
        for row in rows:
            try:
                metadata_obj = json.loads(row.get("metadata") or "{}")
            except Exception:
                metadata_obj = {}
            events.append(f"{row['timestamp']} - {row['event_type']} (Metadata: {metadata_obj}): {row['content']}")
        summary = "\n".join(events)
        logger.info(f"Natural language query fetched events for agent {agent_id}.")
        return summary
    except Exception as e:
        log_exception(e, "Error fetching events for natural language query")
        return "Error retrieving logs."
    finally:
        await release_connection(conn)

# --- Continuous Learning and External Integration ---
async def ingest_external_context(source_url):
    """
    Ingest external data (e.g., news or social media) and return processed context.
    Stub implementation using aiohttp with a timeout.
    """
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(source_url) as resp:
                data = await resp.text()
                # Process the external data as needed. Here, we simply return a summary.
                context = f"Ingested external context: {data[:200]}..."
                logger.info("External context ingested.")
                return context
    except Exception as e:
        log_exception(e, "Error ingesting external context")
        return ""

async def predict_next_event(agent_id):
    """
    Stub for a predictive model that anticipates which events will be most relevant next.
    In a full implementation, this might build a graph of past events and forecast future ones.
    """
    # For demonstration, simply fetch the most recent event and assume similar events will follow.
    recent = await get_recent_events(agent_id, limit=1)
    prediction = f"Predicted next relevant event based on: {recent}"
    logger.info("Predicted next event.")
    return prediction

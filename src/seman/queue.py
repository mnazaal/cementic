"""Persistent SQLite queue for indexing jobs."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional


class JobStatus(str, Enum):
    """Status of a queue job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueueJob:
    """Represents a job in the queue."""

    id: int
    source_path: str
    file_hash: Optional[str]
    status: JobStatus
    priority: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class JobQueue:
    """SQLite-based persistent job queue."""

    def __init__(self, db_path: Path) -> None:
        """Initialize queue with database path."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT,
                    status TEXT DEFAULT 'queued',
                    priority INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queue_status 
                ON queue(status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queue_priority 
                ON queue(priority DESC, created_at)
                """
            )
            conn.commit()

    def add_job(
        self,
        source_path: str,
        file_hash: Optional[str] = None,
        priority: int = 0,
    ) -> bool:
        """Add a job to the queue. Returns True if added, False if already exists."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO queue (source_path, file_hash, status, priority)
                    VALUES (?, ?, 'queued', ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                        file_hash = excluded.file_hash,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE queue.status != 'processing'
                    """,
                    (source_path, file_hash, priority),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_next_job(self) -> Optional[QueueJob]:
        """Get next job to process, marking it as processing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM queue 
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at
                LIMIT 1
                """
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Mark as processing
            conn.execute(
                "UPDATE queue SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (row["id"],),
            )
            conn.commit()

            return QueueJob(
                id=row["id"],
                source_path=row["source_path"],
                file_hash=row["file_hash"],
                status=JobStatus.PROCESSING,
                priority=row["priority"],
                error_message=row["error_message"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def mark_completed(self, job_id: int) -> None:
        """Mark job as completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE queue 
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (job_id,),
            )
            conn.commit()

    def mark_failed(self, job_id: int, error_message: str) -> None:
        """Mark job as failed with error message."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE queue 
                SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (error_message, job_id),
            )
            conn.commit()

    def reset_processing_jobs(self) -> int:
        """Reset any processing jobs back to queued. Returns count reset."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE queue 
                SET status = 'queued', updated_at = CURRENT_TIMESTAMP 
                WHERE status = 'processing'
                """
            )
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT 
                    status,
                    COUNT(*) as count
                FROM queue
                GROUP BY status
                """
            )
            stats = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Ensure all statuses are present
            for status in JobStatus:
                if status.value not in stats:
                    stats[status.value] = 0

            return stats

    def clear_all(self) -> int:
        """Clear all jobs from queue. Returns count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM queue")
            conn.commit()
            return cursor.rowcount

    def get_job_by_path(self, source_path: str) -> Optional[QueueJob]:
        """Get job by source path."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM queue WHERE source_path = ?",
                (source_path,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return QueueJob(
                id=row["id"],
                source_path=row["source_path"],
                file_hash=row["file_hash"],
                status=JobStatus(row["status"]),
                priority=row["priority"],
                error_message=row["error_message"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

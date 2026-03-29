"""Tuning logs API router."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter

router = APIRouter(prefix="/api/tuning", tags=["tuning"])

DB_PATH = "data/cryptotrader.db"


@router.get("")
async def get_tuning_logs(limit: int = 20) -> list[dict]:
    """Return recent auto-tuner logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tuning_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()

        return [
            {
                "id": r["id"],
                "summary": r["summary"],
                "parameters_before": json.loads(r["parameters_before"]),
                "parameters_after": json.loads(r["parameters_after"]),
                "metrics": json.loads(r["metrics"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception:
        return []

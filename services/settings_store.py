"""SQLite-backed settings persistence for ArgueBot."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "arguebot.db"

DEFAULTS: dict[str, Any] = {
    "llm_base_url": "http://127.0.0.1:11434",
    "llm_model": "qwen2.5:7b",
    "llm_temperature": 0.8,
    "llm_max_tokens": 80,
    "chatterbox_base_url": "http://127.0.0.1:8004",
    "chatterbox_voice_mode": "clone",
    "chatterbox_predefined_voice": "Olivia.wav",
    "chatterbox_reference_voice": "kryten2.mp3",
    "chatterbox_temperature": 0.8,
    "chatterbox_exaggeration": 0.5,
    "chatterbox_cfg_weight": 0.5,
    "chatterbox_speed_factor": 1.0,
    "chatterbox_model": "chatterbox-turbo",
    "bot_name": "ArgueBot",
    "bot_persona_blurb": (
        "Unhinged, furious, and barely keeping it together. Snaps at everyone, "
        "rants at the slightest provocation, and sounds like he's about to blow a gasket."
    ),
    "avatar_path": "",
    "rag_top_k": 6,
}


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        for key, value in DEFAULTS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        conn.commit()


def get_settings() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    settings = dict(DEFAULTS)
    for row in rows:
        settings[row["key"]] = json.loads(row["value"])
    return settings


def update_settings(updates: dict[str, Any]) -> dict[str, Any]:
    init_db()
    current = get_settings()
    current.update(updates)
    with _connect() as conn:
        for key, value in updates.items():
            conn.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, json.dumps(value)),
            )
        conn.commit()
    return current


def get_setting(key: str, default: Any = None) -> Any:
    settings = get_settings()
    return settings.get(key, default)
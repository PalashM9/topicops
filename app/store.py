import sqlite3

from app.common import STATE, ensure_dirs

DB = STATE / "predictions.sqlite"


def connect():
    ensure_dirs()
    connection = sqlite3.connect(DB, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize():
    with connect() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, version TEXT NOT NULL,
            label TEXT NOT NULL, confidence REAL NOT NULL, length INTEGER NOT NULL,
            empty_vector INTEGER NOT NULL, latency_ms REAL NOT NULL
        )""")


def record(row):
    with connect() as connection:
        connection.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row)


def recent(version, limit):
    initialize()
    with connect() as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            "SELECT * FROM predictions WHERE version=? ORDER BY timestamp DESC LIMIT ?", (version, limit)
        ).fetchall()]

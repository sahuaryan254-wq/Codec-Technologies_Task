import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'depression_records.db'

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                emotion TEXT,
                score REAL,
                risk_score REAL,
                recorded_at TEXT
            )
            '''
        )
        self.connection.commit()

    def insert_record(self, user_id: str, emotion: str, score: float, risk_score: float) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            'INSERT INTO records (user_id, emotion, score, risk_score, recorded_at) VALUES (?, ?, ?, ?, ?)',
            (user_id, emotion, score, risk_score, datetime.utcnow().isoformat()),
        )
        self.connection.commit()

    def fetch_user_records(self, user_id: str) -> list[tuple]:
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM records WHERE user_id = ? ORDER BY recorded_at DESC', (user_id,))
        return cursor.fetchall()

    def close(self) -> None:
        self.connection.close()

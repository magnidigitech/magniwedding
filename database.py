"""
Database layer for MagniWedding multi-event architecture.
Uses SQLite in WAL mode for microsecond reads and concurrent safety.
"""
import sqlite3
import os
from pathlib import Path

BASE = Path(__file__).parent
STORAGE = Path(os.environ.get("STORAGE_DIR", BASE / "storage"))
DATA_DIR = STORAGE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "magniwedding.db"

def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db()
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            event_type TEXT DEFAULT 'Wedding',
            event_date TEXT,
            location TEXT,
            cover_image_path TEXT,
            description TEXT,
            status TEXT DEFAULT 'published',
            passcode TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS albums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            slug TEXT NOT NULL,
            name TEXT NOT NULL,
            cover_image_path TEXT,
            sort_order INTEGER DEFAULT 0,
            is_published BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(event_id, slug)
        );

        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            album_id INTEGER REFERENCES albums(id) ON DELETE SET NULL,
            filename TEXT NOT NULL,
            rel_path TEXT UNIQUE NOT NULL,
            drive_file_id TEXT,
            file_size_bytes INTEGER DEFAULT 0,
            width INTEGER,
            height INTEGER,
            is_featured BOOLEAN DEFAULT 0,
            is_hidden BOOLEAN DEFAULT 0,
            is_deleted BOOLEAN DEFAULT 0,
            deleted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS person_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            name TEXT DEFAULT 'Unnamed Guest',
            avatar_thumb_path TEXT,
            is_hidden BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS face_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            person_id INTEGER REFERENCES person_profiles(id) ON DELETE SET NULL,
            bbox_json TEXT,
            embedding_offset INTEGER,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_events_slug ON events(slug);
        CREATE INDEX IF NOT EXISTS idx_albums_event ON albums(event_id);
        CREATE INDEX IF NOT EXISTS idx_photos_event_album ON photos(event_id, album_id);
        CREATE INDEX IF NOT EXISTS idx_photos_deleted ON photos(is_deleted);
        """)
    conn.close()

def migrate_existing_data():
    """Migrates existing single-event data into the multi-event structure seamlessly."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if default event exists
    cursor.execute("SELECT id FROM events WHERE slug = 'wedding' OR slug = 'default'")
    row = cursor.fetchone()
    if not row:
        cursor.execute("""
            INSERT INTO events (slug, name, event_type, event_date, location, description, status)
            VALUES ('wedding', 'The Wedding Celebration', 'Wedding', '2026-10-29', 'Hyderabad', 'A beautiful celebration of love and togetherness.', 'published')
        """)
        event_id = cursor.lastrowid
    else:
        event_id = row["id"]
        
    # Check if default album exists
    cursor.execute("SELECT id FROM albums WHERE event_id = ? AND slug = 'ceremony'", (event_id,))
    album_row = cursor.fetchone()
    if not album_row:
        cursor.execute("""
            INSERT INTO albums (event_id, slug, name, sort_order, is_published)
            VALUES (?, 'ceremony', 'Wedding Ceremony', 1, 1)
        """, (event_id,))
        album_id = cursor.lastrowid
    else:
        album_id = album_row["id"]
        
    # Migrate any photos on disk
    photos_dir = STORAGE / "photos"
    data_dir = STORAGE / "data"
    drive_map_file = data_dir / "drive_map.json"
    drive_map = {}
    if drive_map_file.exists():
        import json
        try:
            drive_map = json.loads(drive_map_file.read_text())
        except Exception:
            pass

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if photos_dir.exists():
        for p in photos_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                rel = str(p.relative_to(photos_dir))
                cursor.execute("SELECT id FROM photos WHERE rel_path = ?", (rel,))
                if not cursor.fetchone():
                    file_size = p.stat().st_size
                    drive_id = drive_map.get(rel)
                    cursor.execute("""
                        INSERT OR IGNORE INTO photos (event_id, album_id, filename, rel_path, drive_file_id, file_size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (event_id, album_id, p.name, rel, drive_id, file_size))
                    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    migrate_existing_data()
    print("Database initialized and existing data migrated successfully.")


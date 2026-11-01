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
    try:
        conn.execute("ALTER TABLE albums ADD COLUMN is_deleted BOOLEAN DEFAULT 0;")
        conn.commit()
    except Exception:
        pass
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

        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            task_type TEXT NOT NULL,
            event_slug TEXT NOT NULL,
            target_folder TEXT,
            source_url TEXT,
            status TEXT DEFAULT 'running',
            progress_pct INTEGER DEFAULT 0,
            total_files INTEGER DEFAULT 0,
            processed_files INTEGER DEFAULT 0,
            storage_saved_mb REAL DEFAULT 0,
            current_step TEXT,
            log_text TEXT DEFAULT '',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_events_slug ON events(slug);
        CREATE INDEX IF NOT EXISTS idx_albums_event ON albums(event_id);
        CREATE INDEX IF NOT EXISTS idx_photos_event_album ON photos(event_id, album_id);
        CREATE INDEX IF NOT EXISTS idx_photos_deleted ON photos(is_deleted);
        CREATE INDEX IF NOT EXISTS idx_task_history_id ON task_history(task_id);
        """)
    conn.close()

def create_task_record(task_id: str, task_type: str, event_slug: str, target_folder: str = "", source_url: str = ""):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO task_history (task_id, task_type, event_slug, target_folder, source_url, status, progress_pct, current_step)
        VALUES (?, ?, ?, ?, ?, 'running', 0, 'Starting task...')
    """, (task_id, task_type, event_slug, target_folder, source_url))
    conn.commit()
    conn.close()

def update_task_progress(task_id: str, progress_pct: int = None, current_step: str = None,
                         processed_files: int = None, total_files: int = None,
                         storage_saved_mb: float = None, status: str = None, log_line: str = None):
    conn = get_db()
    c = conn.cursor()
    fields = []
    vals = []
    if progress_pct is not None:
        fields.append("progress_pct = ?")
        vals.append(progress_pct)
    if current_step is not None:
        fields.append("current_step = ?")
        vals.append(current_step)
    if processed_files is not None:
        fields.append("processed_files = ?")
        vals.append(processed_files)
    if total_files is not None:
        fields.append("total_files = ?")
        vals.append(total_files)
    if storage_saved_mb is not None:
        fields.append("storage_saved_mb = ?")
        vals.append(storage_saved_mb)
    if status is not None:
        fields.append("status = ?")
        vals.append(status)
        if status in {"completed", "failed", "cancelled"}:
            fields.append("completed_at = CURRENT_TIMESTAMP")
    if log_line:
        fields.append("log_text = log_text || ?")
        vals.append(f"[{log_line}]\n")
    if fields:
        vals.append(task_id)
        c.execute(f"UPDATE task_history SET {', '.join(fields)} WHERE task_id = ?", vals)
        conn.commit()
    conn.close()

def stop_running_tasks(reason="Task stopped by admin."):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        UPDATE task_history 
        SET status = 'cancelled', 
            current_step = ?, 
            completed_at = CURRENT_TIMESTAMP,
            log_text = log_text || ?
        WHERE status = 'running'
    """, (reason, f"[{reason}]\n"))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected

def get_latest_task():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM task_history ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def list_task_history(limit=50):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM task_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_task_by_id(task_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM task_history WHERE task_id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

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

def cluster_and_sync_people(event_slug="wedding", threshold=0.48):
    """
    Groups indexed face embeddings for an event into accurate Person Profiles.
    Uses centroid-based clustering and merging so the same person is not fragmented,
    and assigns each person a distinct profile and cropped face avatar.
    """
    try:
        import numpy as np
    except ImportError:
        return {"status": "error", "message": "NumPy not available"}

    import json
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM events WHERE slug = ?", (event_slug,))
    event_row = cursor.fetchone()
    if not event_row:
        conn.close()
        return {"status": "error", "message": "Event not found"}
        
    event_id = event_row["id"]
    
    if event_slug in {"default", "wedding"}:
        data_dir = STORAGE / "data"
    else:
        data_dir = STORAGE / "data" / "events" / event_slug
        
    emb_file = data_dir / "embeddings.npy"
    names_file = data_dir / "names.json"
    
    if not emb_file.exists() or not names_file.exists():
        conn.close()
        return {"status": "ok", "people_count": 0, "message": "No embeddings found"}
        
    embs = np.load(emb_file)
    names = json.loads(names_file.read_text())
    
    if len(embs) == 0:
        conn.close()
        return {"status": "ok", "people_count": 0}
        
    # 1. Cosine similarity matrix
    sims = np.dot(embs, embs.T)
    assigned = set()
    initial_clusters = []

    # 2. Centroid-based clustering
    for i in range(len(embs)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        c_emb = embs[i].copy()
        c_emb = c_emb / np.linalg.norm(c_emb)

        # Find all faces matching this cluster centroid
        for j in range(len(embs)):
            if j not in assigned:
                sim_to_c = float(np.dot(embs[j], c_emb))
                if sim_to_c >= threshold:
                    cluster.append(j)
                    assigned.add(j)
                    # Update running centroid
                    c_emb = np.mean(embs[cluster], axis=0)
                    c_emb = c_emb / np.linalg.norm(c_emb)

        initial_clusters.append(cluster)

    # 3. Cluster Merging: merge clusters whose centroids are very close (>= 0.52)
    clusters = []
    centroids = []
    for cl in initial_clusters:
        c_emb = np.mean(embs[cl], axis=0)
        c_emb = c_emb / np.linalg.norm(c_emb)
        merged = False
        for c_idx, existing_c in enumerate(centroids):
            if np.dot(c_emb, existing_c) >= 0.52:
                clusters[c_idx].extend(cl)
                # Re-calculate merged centroid
                new_c = np.mean(embs[clusters[c_idx]], axis=0)
                centroids[c_idx] = new_c / np.linalg.norm(new_c)
                merged = True
                break
        if not merged:
            clusters.append(cl)
            centroids.append(c_emb)

    # Sort clusters by number of unique photo appearances
    clusters.sort(key=lambda c: len(set(names[idx] for idx in c)), reverse=True)
    
    # 4. Preserve existing user-assigned custom names
    cursor.execute("SELECT id, name FROM person_profiles WHERE event_id = ?", (event_id,))
    custom_names = {}
    for r in cursor.fetchall():
        name = r["name"]
        if not name.startswith("Guest #") and name not in {"Featured Person", "Key Family", "Rahul (Groom)", "Priya (Bride)"}:
            custom_names[r["id"]] = name

    # Clear old face_detections and unrenamed profiles for clean sync
    cursor.execute("DELETE FROM face_detections WHERE event_id = ?", (event_id,))
    cursor.execute("DELETE FROM person_profiles WHERE event_id = ?", (event_id,))
    
    # Map photos rel_path, filename and relative subpath to photo id
    cursor.execute("SELECT id, rel_path, filename FROM photos WHERE event_id = ?", (event_id,))
    photo_map = {}
    for r in cursor.fetchall():
        photo_map[r["rel_path"]] = r["id"]
        photo_map[r["filename"]] = r["id"]
        parts = r["rel_path"].split("/")
        if len(parts) >= 2:
            photo_map["/".join(parts[1:])] = r["id"]
        if len(parts) >= 3:
            photo_map["/".join(parts[2:])] = r["id"]
    
    people_created = 0
    for idx, cl in enumerate(clusters):
        sample_raw = names[cl[0]]
        if event_slug not in {"default", "wedding"} and not sample_raw.startswith("events/"):
            sample_rel = f"events/{event_slug}/{sample_raw}"
        else:
            sample_rel = sample_raw
            
        default_name = f"Guest #{idx + 1}"
        if idx == 0:
            default_name = "Rahul (Groom)" if event_slug == "rahul-priya" else "Featured Person"
        elif idx == 1:
            default_name = "Priya (Bride)" if event_slug == "rahul-priya" else "Key Family"
            
        cursor.execute("""
            INSERT INTO person_profiles (event_id, name, avatar_thumb_path)
            VALUES (?, ?, ?)
        """, (event_id, default_name, sample_rel))
        person_id = cursor.lastrowid
        people_created += 1
            
        # Link face detections
        for face_idx in cl:
            photo_rel = names[face_idx]
            fname = Path(photo_rel).name
            p_id = photo_map.get(photo_rel) or photo_map.get(fname)
            if p_id:
                cursor.execute("""
                    INSERT INTO face_detections (event_id, photo_id, person_id, embedding_offset, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (event_id, p_id, person_id, int(face_idx), 0.95))
                
    conn.commit()
    conn.close()

    try:
        from crop_faces import crop_people_faces
        crop_people_faces(event_slug)
    except Exception as e:
        print(f"Face crop note: {e}")

    return {"status": "ok", "people_count": len(clusters), "created": people_created}

def list_people_for_event(event_slug: str):
    """Returns all identified people for an event with thumbnail and photo count."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.name, p.avatar_thumb_path, p.is_hidden,
               COUNT(DISTINCT ph.filename) as photo_count
        FROM person_profiles p
        JOIN events e ON p.event_id = e.id
        LEFT JOIN face_detections fd ON p.id = fd.person_id
        LEFT JOIN photos ph ON fd.photo_id = ph.id AND ph.is_deleted = 0
        WHERE e.slug = ? AND p.is_hidden = 0
        GROUP BY p.id
        HAVING photo_count > 0
        ORDER BY photo_count DESC, p.id ASC
    """, (event_slug,))
    people = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return people

def get_person_photos(event_slug: str, person_id: int):
    """Returns all photos containing a specific person, deduplicated by filename."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ph.id, ph.filename, ph.rel_path, ph.drive_file_id, ph.is_featured,
               a.name as album_name, a.slug as album_slug,
               e.name as event_name, e.slug as event_slug
        FROM photos ph
        JOIN face_detections fd ON ph.id = fd.photo_id
        JOIN events e ON ph.event_id = e.id
        LEFT JOIN albums a ON ph.album_id = a.id
        WHERE e.slug = ? AND fd.person_id = ? AND ph.is_deleted = 0 AND ph.is_hidden = 0
        GROUP BY ph.filename
        ORDER BY ph.is_featured DESC, ph.filename ASC
    """, (event_slug, person_id))
    photos = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return photos

if __name__ == "__main__":
    init_db()
    migrate_existing_data()
    print("Database initialized and existing data migrated successfully.")


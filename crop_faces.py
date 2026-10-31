import sqlite3
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import os
import json

STORAGE = Path(os.environ.get("STORAGE_DIR", "/storage"))
MODELS = STORAGE / "models"
DATA = STORAGE / "data"
PHOTOS = STORAGE / "photos"
FACES = DATA / "faces"
FACES.mkdir(parents=True, exist_ok=True)

def crop_people_faces(target_event_slug=None):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=str(MODELS), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(960, 960))

    conn = sqlite3.connect(str(DATA / "magniwedding.db"))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if target_event_slug:
        c.execute("""
            SELECT p.id, p.event_id, p.name, p.avatar_thumb_path, e.slug as event_slug 
            FROM person_profiles p
            JOIN events e ON p.event_id = e.id
            WHERE e.slug = ?
        """, (target_event_slug,))
    else:
        c.execute("""
            SELECT p.id, p.event_id, p.name, p.avatar_thumb_path, e.slug as event_slug 
            FROM person_profiles p
            JOIN events e ON p.event_id = e.id
        """)
    profiles = c.fetchall()
    print(f"Processing {len(profiles)} profiles for accurate face avatars...")

    cache_img_faces = {}
    event_embs_cache = {}

    for p in profiles:
        pid = p["id"]
        ev_slug = p["event_slug"]

        # Load event embeddings if not already in memory
        if ev_slug not in event_embs_cache:
            if ev_slug in {"default", "wedding"}:
                e_path = DATA / "embeddings.npy"
            else:
                e_path = DATA / "events" / ev_slug / "embeddings.npy"
            if e_path.exists():
                event_embs_cache[ev_slug] = np.load(e_path)
            else:
                event_embs_cache[ev_slug] = None

        embs = event_embs_cache.get(ev_slug)

        # Get all detections for this person
        c.execute("""
            SELECT fd.embedding_offset, ph.rel_path, ph.filename
            FROM face_detections fd
            JOIN photos ph ON fd.photo_id = ph.id
            WHERE fd.person_id = ?
        """, (pid,))
        detections = c.fetchall()
        if not detections:
            continue

        # Compute centroid embedding of this person
        person_emb = None
        if embs is not None:
            offsets = [d["embedding_offset"] for d in detections if d["embedding_offset"] < len(embs)]
            if offsets:
                p_embs = embs[offsets]
                mean_e = np.mean(p_embs, axis=0)
                person_emb = mean_e / np.linalg.norm(mean_e)

        # Find best photo to crop from
        best_crop_face = None
        best_crop_img = None
        best_sim = -1.0

        for d in detections[:6]: # Test top candidate photos
            rel = d["rel_path"]
            img_path = PHOTOS / rel
            if not img_path.exists():
                matches = list(PHOTOS.glob(f"**/{Path(rel).name}"))
                if matches:
                    img_path = matches[0]
                else:
                    continue

            cache_key = str(img_path)
            if cache_key not in cache_img_faces:
                try:
                    img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
                    img.thumbnail((2000, 2000))
                    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    faces = app.get(bgr)
                    cache_img_faces[cache_key] = (img, faces)
                except Exception:
                    continue
            else:
                img, faces = cache_img_faces[cache_key]

            if not faces:
                continue

            # CRITICAL FIX: Match face in photo to this person's centroid embedding!
            if person_emb is not None:
                for f in faces:
                    sim = float(np.dot(f.normed_embedding, person_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_crop_face = f
                        best_crop_img = img
            else:
                best_crop_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                best_crop_img = img
                break

            if best_sim >= 0.65:
                break # High quality face found, no need to check more photos

        if best_crop_face is None or best_crop_img is None:
            continue

        face = best_crop_face
        img = best_crop_img
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * 0.35)
        pad_y = int(h * 0.35)

        box = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img.width, x2 + pad_x),
            min(img.height, y2 + pad_y)
        )
        face_img = img.crop(box)
        face_img.thumbnail((240, 240))

        out_file = FACES / f"person_{pid}.jpg"
        face_img.save(out_file, quality=92)

        new_rel = f"faces/person_{pid}.jpg"
        c.execute("UPDATE person_profiles SET avatar_thumb_path = ? WHERE id = ?", (new_rel, pid))

    conn.commit()
    conn.close()
    saved_count = len(list(FACES.glob("*.jpg")))
    print(f"Done! {saved_count} accurate face avatars ready in {FACES}.")

if __name__ == "__main__":
    crop_people_faces()


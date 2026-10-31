import sqlite3
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import os

STORAGE = Path(os.environ.get("STORAGE_DIR", "/storage"))
MODELS = STORAGE / "models"
DATA = STORAGE / "data"
PHOTOS = STORAGE / "photos"
FACES = DATA / "faces"
FACES.mkdir(parents=True, exist_ok=True)

def crop_people_faces():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=str(MODELS), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(960, 960))

    conn = sqlite3.connect(str(DATA / "magniwedding.db"))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT id, event_id, name, avatar_thumb_path FROM person_profiles")
    profiles = c.fetchall()
    print(f"Processing {len(profiles)} profiles for face avatars...")

    cache_crops = {}

    for p in profiles:
        pid = p["id"]
        rel = p["avatar_thumb_path"]
        if not rel:
            continue
        # Clean relative path if already has faces/
        if rel.startswith("faces/"):
            out_file = FACES / f"person_{pid}.jpg"
            if out_file.exists():
                continue
            rel = rel.replace("faces/", "")
            
        img_path = PHOTOS / rel
        if not img_path.exists():
            # Check with subfolders
            matches = list(PHOTOS.glob(f"**/{Path(rel).name}"))
            if matches:
                img_path = matches[0]
            else:
                continue

        cache_key = str(img_path)
        if cache_key not in cache_crops:
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            img.thumbnail((2000, 2000))
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            cache_crops[cache_key] = (img, faces)
        else:
            img, faces = cache_crops[cache_key]

        if not faces:
            continue

        # Sort faces by detection box size
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
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
    print(f"Done! {saved_count} face avatars ready in {FACES}.")

if __name__ == "__main__":
    crop_people_faces()

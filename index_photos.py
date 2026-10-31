"""Run once (and again if you add photos): finds every face in every photo,
saves face embeddings + small thumbnails. Safe to stop and re-run; it resumes."""
import json, os, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

BASE = Path(__file__).parent
STORAGE = Path(os.environ.get("STORAGE_DIR", BASE / "storage"))
PHOTOS = STORAGE / "photos"
DATA = STORAGE / "data"
MODELS = STORAGE / "models"
THUMBS = DATA / "thumbs"
DET_SIZE = int(os.environ.get("DET_SIZE", "960"))   # bigger = finds smaller faces, slower
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DATA.mkdir(exist_ok=True)
THUMBS.mkdir(exist_ok=True)

emb_file, names_file, done_file = DATA / "embeddings.npy", DATA / "names.json", DATA / "done.json"
embs = list(np.load(emb_file)) if emb_file.exists() else []
names = json.loads(names_file.read_text()) if names_file.exists() else []
done = set(json.loads(done_file.read_text())) if done_file.exists() else set()

def save():
    np.save(emb_file, np.array(embs, dtype=np.float32).reshape(-1, 512))
    names_file.write_text(json.dumps(names))
    done_file.write_text(json.dumps(sorted(done)))

face_app = FaceAnalysis(name="buffalo_l", root=str(MODELS), providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))

files = sorted(p for p in PHOTOS.rglob("*") if p.suffix.lower() in EXTS)
print(f"Found {len(files)} photos, {len(done)} already done.")

for i, path in enumerate(files, 1):
    rel = str(path.relative_to(PHOTOS))
    if rel in done:
        continue
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((480, 480))
        thumb.save(THUMBS / (rel.replace("/", "__") + ".jpg"), quality=80)

        img.thumbnail((2000, 2000))
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for face in face_app.get(bgr):
            embs.append(face.normed_embedding)
            names.append(rel)
    except Exception as e:
        print(f"  skipped {rel}: {e}")
    done.add(rel)
    if i % 50 == 0:
        print(f"{i}/{len(files)}  faces so far: {len(embs)}")
        save()

save()
print(f"Done. {len(done)} photos, {len(embs)} faces indexed.")

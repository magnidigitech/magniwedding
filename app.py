import os, threading
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
import json

BASE = Path(__file__).parent
STORAGE = Path(os.environ.get("STORAGE_DIR", BASE / "storage"))
PHOTOS = STORAGE / "photos"
DATA = STORAGE / "data"
MODELS = STORAGE / "models"
MODELS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))  # raise = stricter
MAX_RESULTS = 300
MAX_UPLOAD = 10 * 1024 * 1024

embs = np.zeros((0, 512), dtype=np.float32)
names, known, loaded_mtime = [], set(), 0.0

def reload_index():
    """Re-reads the index whenever indexing has produced a newer file (no restart needed)."""
    global embs, names, known, loaded_mtime
    f = DATA / "embeddings.npy"
    if f.exists() and f.stat().st_mtime != loaded_mtime:
        embs = np.load(f)
        names = json.loads((DATA / "names.json").read_text())
        known = set(names)
        loaded_mtime = f.stat().st_mtime

reload_index()

face_app = FaceAnalysis(name="buffalo_l", root=str(MODELS), providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
lock = threading.Lock()  # one search at a time keeps the CPU happy

app = FastAPI()

@app.post("/api/search")
def search(selfie: UploadFile = File(...)):
    reload_index()
    if len(names) == 0:
        raise HTTPException(503, "Photos are still being prepared. Please check back soon.")
    data = selfie.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Photo is too large. Use one under 10 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # selfie never saved to disk
    if img is None:
        raise HTTPException(400, "That file is not a readable image.")
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        s = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    with lock:
        faces = face_app.get(img)
    if not faces:
        raise HTTPException(422, "No face found. Try a clear, front-facing photo in good light.")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    sims = embs @ face.normed_embedding
    best = {}
    for idx in np.where(sims >= THRESHOLD)[0]:
        n = names[idx]
        best[n] = max(best.get(n, 0), float(sims[idx]))
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:MAX_RESULTS]
    return {"count": len(ranked), "photos": [n for n, _ in ranked]}

@app.get("/thumb/{name:path}")
def thumb(name: str):
    if name not in known:
        raise HTTPException(404)
    return FileResponse(DATA / "thumbs" / (name.replace("/", "__") + ".jpg"))

@app.get("/photo/{name:path}")
def photo(name: str):
    if name not in known:
        raise HTTPException(404)
    return FileResponse(PHOTOS / name, filename=Path(name).name)

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")

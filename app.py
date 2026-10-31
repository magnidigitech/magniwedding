import os, sys, threading, subprocess, secrets, shutil, zipfile, io
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Depends, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
import json

BASE = Path(__file__).parent
STORAGE = Path(os.environ.get("STORAGE_DIR", BASE / "storage"))
PHOTOS = STORAGE / "photos"
DATA = STORAGE / "data"
MODELS = STORAGE / "models"
PHOTOS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))  # raise = stricter
MAX_RESULTS = 300
MAX_UPLOAD = 10 * 1024 * 1024
ADMIN_KEY = os.environ.get("ADMIN_KEY", "Magni@3939")

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

# Admin session & background tasks
active_tokens = set()
task_lock = threading.Lock()
current_task = {"status": "idle", "type": None, "message": "Ready"}
log_buffer = []

def append_log(msg: str):
    global log_buffer
    log_buffer.append(msg)
    if len(log_buffer) > 500:
        log_buffer.pop(0)
    try:
        with open(STORAGE / "index.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def get_auth_token(req: Request) -> str:
    token = req.headers.get("X-Admin-Token") or req.cookies.get("admin_token")
    if not token or token not in active_tokens:
        raise HTTPException(401, "Unauthorized. Please log in with the admin key.")
    return token

def run_indexing_task():
    global current_task
    with task_lock:
        current_task["status"] = "running"
        current_task["type"] = "indexing"
        current_task["message"] = "Face indexing in progress..."
    append_log("--- Starting Face Indexing ---")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(BASE / "index_photos.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE)
        )
        for line in proc.stdout:
            append_log(line.rstrip())
        proc.wait()
        reload_index()
        append_log(f"--- Face Indexing Completed (exit code {proc.returncode}) ---")
        with task_lock:
            current_task["status"] = "idle"
            current_task["type"] = None
            current_task["message"] = f"Indexing finished successfully ({len(known)} photos, {len(embs)} faces)."
    except Exception as e:
        append_log(f"Indexing error: {e}")
        with task_lock:
            current_task["status"] = "error"
            current_task["type"] = None
            current_task["message"] = f"Error: {e}"

def run_gdrive_download(url: str, folder_name: str):
    global current_task
    with task_lock:
        current_task["status"] = "running"
        current_task["type"] = "gdrive"
        current_task["message"] = f"Downloading from Google Drive into '{folder_name}'..."
    append_log(f"--- Starting Google Drive Download into {folder_name} ---")
    dest_dir = PHOTOS / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "gdown", "--folder", url, "-O", str(dest_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            append_log(line.rstrip())
        proc.wait()
        append_log(f"--- Google Drive Download Finished (exit code {proc.returncode}) ---")
        # Automatically run indexing after download completes
        run_indexing_task()
    except Exception as e:
        append_log(f"Google Drive download error: {e}")
        with task_lock:
            current_task["status"] = "error"
            current_task["type"] = None
            current_task["message"] = f"Error: {e}"

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
        reload_index()
    if name not in known:
        raise HTTPException(404)
    return FileResponse(DATA / "thumbs" / (name.replace("/", "__") + ".jpg"))

@app.get("/photo/{name:path}")
def photo(name: str):
    if name not in known:
        reload_index()
    if name not in known:
        raise HTTPException(404)
    return FileResponse(PHOTOS / name, filename=Path(name).name)

# --- Admin APIs ---

@app.post("/api/admin/login")
async def admin_login(payload: dict):
    key = payload.get("key", "").strip()
    if key != ADMIN_KEY:
        raise HTTPException(401, "Incorrect admin key. Please try again.")
    token = secrets.token_hex(24)
    active_tokens.add(token)
    response = JSONResponse({"status": "ok", "token": token})
    response.set_cookie("admin_token", token, httponly=True, samesite="lax", max_age=86400 * 7)
    return response

@app.post("/api/admin/logout")
def admin_logout(token: str = Depends(get_auth_token)):
    active_tokens.discard(token)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("admin_token")
    return response

@app.get("/api/admin/status")
def admin_status(token: str = Depends(get_auth_token)):
    reload_index()
    folders = []
    total_files = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if PHOTOS.exists():
        for p in sorted(PHOTOS.iterdir()):
            if p.is_dir():
                c = sum(1 for f in p.rglob("*") if f.suffix.lower() in valid_exts)
                folders.append({"name": p.name, "count": c})
                total_files += c
            elif p.suffix.lower() in valid_exts:
                total_files += 1
    return {
        "status": "ok",
        "total_photos": total_files,
        "indexed_photos": len(known),
        "indexed_faces": len(names),
        "folders": folders,
        "current_task": current_task
    }

@app.post("/api/admin/upload")
async def admin_upload(
    files: list[UploadFile] = File(...),
    folder: str = Form("uploads"),
    auto_index: bool = Form(True),
    token: str = Depends(get_auth_token)
):
    safe_folder = "".join(c for c in folder.strip() if c.isalnum() or c in "-_") or "uploads"
    target_dir = PHOTOS / safe_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    for f in files:
        filename = Path(f.filename or "upload").name
        ext = Path(filename).suffix.lower()
        if ext == ".zip":
            content = await f.read()
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    for info in z.infolist():
                        if info.is_dir() or ".." in info.filename or info.filename.startswith(("/", "\\")):
                            continue
                        sub_ext = Path(info.filename).suffix.lower()
                        if sub_ext in valid_exts:
                            sub_name = Path(info.filename).name
                            dest_path = target_dir / sub_name
                            with z.open(info) as src, open(dest_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            saved_count += 1
            except Exception as e:
                append_log(f"Error extracting {filename}: {e}")
        elif ext in valid_exts:
            content = await f.read()
            dest_path = target_dir / filename
            with open(dest_path, "wb") as dst:
                dst.write(content)
            saved_count += 1
            
    append_log(f"Uploaded {saved_count} photos to '{safe_folder}'.")
    if auto_index and saved_count > 0 and current_task["status"] != "running":
        threading.Thread(target=run_indexing_task, daemon=True).start()
        
    return {"status": "ok", "saved_count": saved_count, "folder": safe_folder}

@app.post("/api/admin/gdrive")
def admin_gdrive(payload: dict, token: str = Depends(get_auth_token)):
    if current_task["status"] == "running":
        raise HTTPException(409, "A task is already running. Please wait for it to finish.")
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(400, "Google Drive URL is required.")
    folder = payload.get("folder", "").strip() or "gdrive_photos"
    safe_folder = "".join(c for c in folder if c.isalnum() or c in "-_") or "gdrive_photos"
    threading.Thread(target=run_gdrive_download, args=(url, safe_folder), daemon=True).start()
    return {"status": "started", "folder": safe_folder}

@app.post("/api/admin/index")
def admin_trigger_index(token: str = Depends(get_auth_token)):
    if current_task["status"] == "running":
        raise HTTPException(409, "A task is already running. Please wait for it to finish.")
    threading.Thread(target=run_indexing_task, daemon=True).start()
    return {"status": "started"}

@app.get("/api/admin/logs")
def admin_logs(token: str = Depends(get_auth_token)):
    log_file = STORAGE / "index.log"
    lines = []
    if log_file.exists():
        try:
            content = log_file.read_text(encoding="utf-8", errors="ignore")
            lines = content.strip().splitlines()[-100:]
        except Exception:
            lines = log_buffer[-100:]
    else:
        lines = log_buffer[-100:]
    return {"logs": lines, "task": current_task}

@app.get("/admin")
def admin_page():
    return FileResponse(BASE / "static" / "admin.html")

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")

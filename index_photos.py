"""
Event-aware Face Indexer for MagniWedding.
Can index a specific event (--event <slug>) or all events.
Saves per-event face embeddings, names, and thumbnails.
"""
import argparse, json, os, sys
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
DET_SIZE = int(os.environ.get("DET_SIZE", "960"))
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def index_event(event_slug="wedding"):
    print(f"=== Starting Face Indexing for Event: '{event_slug}' ===")
    
    # Per-event directories
    if event_slug in {"default", "wedding"}:
        event_photos_dir = PHOTOS
        event_data_dir = DATA
    else:
        event_photos_dir = PHOTOS / "events" / event_slug
        event_data_dir = DATA / "events" / event_slug
        
    event_photos_dir.mkdir(parents=True, exist_ok=True)
    event_data_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = event_data_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    emb_file = event_data_dir / "embeddings.npy"
    names_file = event_data_dir / "names.json"
    done_file = event_data_dir / "done.json"

    embs = list(np.load(emb_file)) if emb_file.exists() else []
    names = json.loads(names_file.read_text()) if names_file.exists() else []
    done = set(json.loads(done_file.read_text())) if done_file.exists() else set()

    def save():
        if embs:
            np.save(emb_file, np.array(embs, dtype=np.float32).reshape(-1, 512))
        else:
            np.save(emb_file, np.zeros((0, 512), dtype=np.float32))
        names_file.write_text(json.dumps(names))
        done_file.write_text(json.dumps(sorted(done)))
        
        # Mirror to root DATA for backward compatibility if default event
        if event_slug in {"default", "wedding"}:
            if embs:
                np.save(DATA / "embeddings.npy", np.array(embs, dtype=np.float32).reshape(-1, 512))
            (DATA / "names.json").write_text(json.dumps(names))
            (DATA / "done.json").write_text(json.dumps(sorted(done)))

    face_app = FaceAnalysis(name="buffalo_l", root=str(MODELS), providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))

    files = sorted(p for p in event_photos_dir.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("."))
    print(f"Found {len(files)} photos in {event_photos_dir}, {len(done)} already indexed.")

    for i, path in enumerate(files, 1):
        rel = str(path.relative_to(event_photos_dir))
        if rel in done:
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            
            # Save thumbnail
            thumb = img.copy()
            thumb.thumbnail((480, 480))
            thumb_name = rel.replace("/", "__") + ".jpg"
            thumb.save(thumbs_dir / thumb_name, quality=80)
            if event_slug in {"default", "wedding"}:
                (DATA / "thumbs").mkdir(parents=True, exist_ok=True)
                thumb.save(DATA / "thumbs" / thumb_name, quality=80)

            # Detect faces
            img.thumbnail((2000, 2000))
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for face in face_app.get(bgr):
                embs.append(face.normed_embedding)
                names.append(rel)
        except Exception as e:
            print(f"  Skipped {rel}: {e}")
            
        done.add(rel)
        if i % 25 == 0:
            print(f"Progress: {i}/{len(files)} photos | Faces so far: {len(embs)}")
            save()

    save()
    print(f"Done for '{event_slug}'. {len(done)} photos, {len(embs)} faces indexed.")
    try:
        from database import cluster_and_sync_people
        cl_res = cluster_and_sync_people(event_slug)
        print(f"Clustering complete: {cl_res}")
    except Exception as e:
        print(f"Clustering note: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MagniWedding Multi-Event Face Indexer")
    parser.add_argument("--event", type=str, default="wedding", help="Event slug to index")
    args = parser.parse_args()
    index_event(args.event)

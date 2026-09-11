import os
import cv2
import json
import base64
import asyncio
import threading
import traceback
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mapping import MEDIAPIPE_TO_ANIME_INDICES

DEFAULT_ALPHA = 0.3  # Alpha = Expression weight; (1 - Alpha) = Spatial weight

anime_filenames = []
matrix_spatial = None
matrix_expr = None
landmarker = None
is_ready = False


def get_image_b64(filename):
    """Dynamically fetch and encode image from disk to avoid RAM OOM crash."""
    if not filename:
        return ""
    img_path = os.path.join("anime_images", filename)
    if not os.path.exists(img_path):
        return ""
    img = cv2.imread(img_path)
    if img is None:
        return ""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def load_database():
    global matrix_spatial, matrix_expr, landmarker, is_ready
    print("Loading database in background...")

    try:
        if not os.path.exists("anime_landmarks_db.npy"):
            print("ERROR: anime_landmarks_db.npy not found!")
            return

        raw_db = np.load("anime_landmarks_db.npy", allow_pickle=True).item()
        print(f"Loaded raw_db with {len(raw_db)} entries.")

        anime_spatial_vectors = []
        anime_expr_vectors = []
        image_dir = "anime_images"

        if not os.path.exists(image_dir):
            print(f"WARNING: Directory '{image_dir}' does not exist.")

        loaded_count = 0
        w_ref, h_ref = 1920.0, 1080.0  # Default normalization reference frame

        for filename, points in raw_db.items():
            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                continue

            pts = np.array(points, dtype=np.float32)
            centroid_raw = np.mean(pts, axis=0)
            centered_raw = pts - centroid_raw
            expr_scale = np.linalg.norm(centered_raw)

            expr_vec = (
                (centered_raw / expr_scale).flatten()
                if expr_scale > 0
                else centered_raw.flatten()
            )
            spatial_vec = np.array(
                [centroid_raw[0] / w_ref, centroid_raw[1] / h_ref, expr_scale / w_ref]
            )

            anime_filenames.append(filename)
            anime_spatial_vectors.append(spatial_vec)
            anime_expr_vectors.append(expr_vec)
            loaded_count += 1

        print(
            f"Successfully processed {loaded_count} image vectors for feature matrices."
        )

        matrix_spatial = np.array(anime_spatial_vectors)
        matrix_expr = np.array(anime_expr_vectors)

        if not os.path.exists("face_landmarker.task"):
            print("ERROR: face_landmarker.task not found!")
            return

        print("Initializing MediaPipe Face Landmarker...")
        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task"
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        is_ready = True
        print("Startup complete. Backend ready.")

    except Exception as e:
        print(f"Fatal error during load_database execution: {e}")
        traceback.print_exc()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_database)
    thread.daemon = True
    thread.start()


@app.get("/")
@app.head("/")
async def get_index():
    return FileResponse("index.html")


@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("favicon.ico")


@app.get("/loading.gif")
async def get_loading():
    return FileResponse("loading.gif")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message_text = await websocket.receive_text()

            if not is_ready:
                await websocket.send_json({"status": "loading"})
                await asyncio.sleep(0.2)
                continue

            image_data = ""
            alpha = DEFAULT_ALPHA

            if message_text.startswith("{"):
                try:
                    payload = json.loads(message_text)
                    image_data = payload.get("image", "")
                    alpha = float(payload.get("alpha", DEFAULT_ALPHA))
                except Exception:
                    continue
            else:
                image_data = message_text

            if not image_data.startswith("data:image"):
                continue

            try:
                header, encoded = image_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            f_h, f_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            best_filename = None
            landmarks_list = []

            if result.face_landmarks:
                face = result.face_landmarks[0]
                h_canvas = []
                for idx in MEDIAPIPE_TO_ANIME_INDICES:
                    nx, ny = face[idx].x, face[idx].y
                    landmarks_list.append({"x": nx, "y": ny})
                    h_canvas.append([nx * f_w, ny * f_h])

                h_pts = np.array(h_canvas, dtype=np.float32)
                h_centroid = np.mean(h_pts, axis=0)
                h_centered = h_pts - h_centroid
                h_expr_scale = np.linalg.norm(h_centered)
                h_expr_vec = (
                    (h_centered / h_expr_scale).flatten()
                    if h_expr_scale > 0
                    else h_centered.flatten()
                )

                h_spatial_vec = np.array([
                    h_centroid[0] / f_w,
                    h_centroid[1] / f_h,
                    h_expr_scale / f_w,
                ])

                dist_expr = np.linalg.norm(matrix_expr - h_expr_vec, axis=1)
                dist_spatial = np.linalg.norm(
                    matrix_spatial - h_spatial_vec, axis=1
                )

                # Normalized linear blend between expression and spatial distance
                w_expr = alpha
                w_spatial = 1.0 - alpha
                total_distances = (w_expr * dist_expr) + (
                    w_spatial * dist_spatial
                )

                best_match_idx = np.argmin(total_distances)
                best_filename = anime_filenames[best_match_idx]

            matched_image_b64 = get_image_b64(best_filename)

            await websocket.send_json({
                "status": "ready",
                "matched_image": matched_image_b64,
                "landmarks": landmarks_list,
            })

    except WebSocketDisconnect:
        pass

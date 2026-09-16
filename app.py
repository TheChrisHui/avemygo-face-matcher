import os
import json
import base64
import asyncio
import threading
import traceback
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mapping import MEDIAPIPE_TO_ANIME_INDICES

DEFAULT_ALPHA = 0.3  # Alpha = Expression weight; (1 - Alpha) = Spatial weight
DEFAULT_IMAGE_W, DEFAULT_IMAGE_H = 1280, 720
MAX_DETECT_WIDTH = 640

anime_filenames = []
anime_image_dims = []
matrix_spatial = None
matrix_expr = None
landmarker = None
is_ready = False
num_entries = 0
load_error = None
frames_received = 0
faces_detected = 0


def load_database():
    global matrix_spatial, matrix_expr, landmarker, is_ready, num_entries, load_error
    load_error = None
    print("Loading database in background...")

    try:
        if not os.path.exists("anime_landmarks_db.npy"):
            load_error = "anime_landmarks_db.npy not found!"
            print(f"ERROR: {load_error}")
            return

        raw_db = np.load("anime_landmarks_db.npy", allow_pickle=True).item()
        print(f"Loaded raw_db with {len(raw_db)} entries.")

        anime_spatial_vectors = []
        anime_expr_vectors = []

        # Compute feature matrices purely from stored landmarks in memory
        for filename, points in raw_db.items():
            if isinstance(points, dict):
                pts_data = points.get("points", [])
                w = float(points.get("w", DEFAULT_IMAGE_W))
                h = float(points.get("h", DEFAULT_IMAGE_H))
            else:
                pts_data = points
                w, h = float(DEFAULT_IMAGE_W), float(DEFAULT_IMAGE_H)

            pts = np.array(pts_data, dtype=np.float32)
            if len(pts) < 2:
                continue

            centroid_raw = np.mean(pts, axis=0)
            centered_raw = pts - centroid_raw
            expr_scale = np.linalg.norm(centered_raw)

            expr_vec = (
                (centered_raw / expr_scale).flatten()
                if expr_scale > 0
                else centered_raw.flatten()
            )

            # Spatial vector normalized by the image the landmarks came from:
            # [Center X%, Center Y%, Face Width %]
            spatial_vec = np.array(
                [centroid_raw[0] / w, centroid_raw[1] / h, expr_scale / w]
            )

            anime_filenames.append(filename)
            anime_image_dims.append((w, h))
            anime_spatial_vectors.append(spatial_vec)
            anime_expr_vectors.append(expr_vec)

        num_entries = len(anime_filenames)
        print(f"Successfully processed {num_entries} landmark entries.")

        if num_entries == 0:
            load_error = "No landmark entries could be processed."
            print(f"ERROR: {load_error}")
            return

        matrix_spatial = np.array(anime_spatial_vectors)
        matrix_expr = np.array(anime_expr_vectors)
        del raw_db

        if not os.path.exists("face_landmarker.task"):
            load_error = "face_landmarker.task not found!"
            print(f"ERROR: {load_error}")
            return

        print("Initializing MediaPipe Face Landmarker (CPU Mode)...")
        # Explicit CPU delegate bypasses GPU/GLES library loading
        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task",
            delegate=python.BaseOptions.Delegate.CPU,
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
        load_error = traceback.format_exc()
        print(f"Fatal error during load_database execution: {e}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_database)
    thread.daemon = True
    thread.start()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/anime_images", StaticFiles(directory="anime_images"), name="anime_images")


@app.get("/status")
async def get_status():
    return {
        "is_ready": is_ready,
        "entries": num_entries,
        "error": load_error,
        "frames_received": frames_received,
        "faces_detected": faces_detected,
    }


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
    global frames_received, faces_detected
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

            frames_received += 1

            # Downscale for CPU: landmarks are normalized, so the math is unchanged
            f_h, f_w, _ = frame.shape
            if f_w > MAX_DETECT_WIDTH:
                scale = MAX_DETECT_WIDTH / f_w
                frame = cv2.resize(
                    frame,
                    (MAX_DETECT_WIDTH, max(1, int(f_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            f_h, f_w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )

            result = await asyncio.to_thread(landmarker.detect, mp_image)

            if result.face_landmarks:
                faces_detected += 1

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

                w_expr = alpha
                w_spatial = 1.0 - alpha
                total_distances = (w_expr * dist_expr) + (
                    w_spatial * dist_spatial
                )

                best_match_idx = int(np.argmin(total_distances))
                best_filename = anime_filenames[best_match_idx]

            matched_image_url = (
                f"/anime_images/{best_filename}" if best_filename else ""
            )

            await websocket.send_json({
                "status": "ready",
                "matched_image_url": matched_image_url,
                "landmarks": landmarks_list,
            })

    except WebSocketDisconnect:
        pass
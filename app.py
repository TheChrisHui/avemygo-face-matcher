import base64
import os
import threading
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matrix_spatial = None
matrix_expr = None
anime_filenames = []
landmarker = None
is_ready = False


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

        for filename, points in raw_db.items():
            img_path = os.path.join(image_dir, filename)
            h, w = 1080, 1920

            # Get image dimensions without storing heavy base64 strings in memory
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]

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
                [centroid_raw[0] / w, centroid_raw[1] / h, expr_scale / w]
            )

            anime_filenames.append(filename)
            anime_spatial_vectors.append(spatial_vec)
            anime_expr_vectors.append(expr_vec)

        matrix_spatial = np.array(anime_spatial_vectors)
        matrix_expr = np.array(anime_expr_vectors)

        if not os.path.exists("face_landmarker.task"):
            print("ERROR: face_landmarker.task not found!")
            return

        print("Initializing MediaPipe Face Landmarker...")
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
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


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_database)
    thread.daemon = True
    thread.start()


@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.get("/loading.gif")
def get_loading_gif():
    return FileResponse("loading.gif")


@app.get("/favicon.ico")
def get_favicon():
    return FileResponse("favicon.ico")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            data = await websocket.receive_json()

            if not is_ready:
                await websocket.send_json({"status": "loading"})
                continue

            image_data = data.get("image", "")
            alpha = float(data.get("alpha", 0.3))

            if not image_data.startswith("data:image"):
                await websocket.send_json({"status": "error", "message": "Invalid image format"})
                continue

            encoded_data = image_data.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"status": "error", "message": "Decode failed"})
                continue

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                await websocket.send_json({"landmarks": None, "matched_image": None})
                continue

            user_landmarks = detection_result.face_landmarks[0]
            pts = np.array([[pt.x * w, pt.y * h] for pt in user_landmarks], dtype=np.float32)

            centroid_raw = np.mean(pts, axis=0)
            centered_raw = pts - centroid_raw
            expr_scale = np.linalg.norm(centered_raw)

            user_expr_vec = (
                (centered_raw / expr_scale).flatten()
                if expr_scale > 0
                else centered_raw.flatten()
            )
            user_spatial_vec = np.array(
                [centroid_raw[0] / w, centroid_raw[1] / h, expr_scale / w]
            )

            dist_spatial = np.linalg.norm(matrix_spatial - user_spatial_vec, axis=1)
            dist_expr = np.linalg.norm(matrix_expr - user_expr_vec, axis=1)

            norm_spatial = (
                dist_spatial / np.max(dist_spatial)
                if np.max(dist_spatial) > 0
                else dist_spatial
            )
            norm_expr = (
                dist_expr / np.max(dist_expr)
                if np.max(dist_expr) > 0
                else dist_expr
            )

            combined_dist = (1 - alpha) * norm_spatial + alpha * norm_expr
            best_idx = int(np.argmin(combined_dist))
            best_filename = anime_filenames[best_idx]

            # Dynamically read and base64-encode only the matched image
            matched_image_b64 = ""
            matched_path = os.path.join("anime_images", best_filename)
            if os.path.exists(matched_path):
                img = cv2.imread(matched_path)
                if img is not None:
                    _, buffer = cv2.imencode(".jpg", img)
                    matched_image_b64 = base64.b64encode(buffer).decode("utf-8")

            landmark_pts = [{"x": pt.x, "y": pt.y} for pt in user_landmarks]

            await websocket.send_json({
                "status": "success",
                "landmarks": landmark_pts,
                "matched_image": matched_image_b64,
            })

        except WebSocketDisconnect:
            break
        except Exception as e:
            print(f"Error in websocket loop: {e}")
            break

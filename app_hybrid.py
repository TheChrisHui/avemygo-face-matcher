import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mapping import MEDIAPIPE_TO_ANIME_INDICES

# ==========================================
# THE TUNING KNOBS
# ==========================================
WEIGHT_EXPRESSION = 1.0
WEIGHT_SPATIAL = 0.3  

def resize_maintain_aspect(img, target_height=400):
    h, w = img.shape[:2]
    aspect = w / h
    new_w = int(target_height * aspect)
    return cv2.resize(img, (new_w, target_height))

# ==========================================
# 1. DATABASE PREPARATION (Aspect-Corrected)
# ==========================================
print("Loading Hybrid Database with Canonical Mapping...")
raw_db = np.load('anime_landmarks_db.npy', allow_pickle=True).item()

anime_filenames = []
anime_spatial_vectors = []
anime_expr_vectors = []
image_dir = 'anime_images'

for filename, points in raw_db.items():
    img_path = os.path.join(image_dir, filename)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        pts = np.array(points, dtype=np.float32)
        
        # 1. Canonical Square Normalization
        s = max(h, w)
        canvas_pts = np.zeros_like(pts)
        canvas_pts[:, 0] = (pts[:, 0] - w / 2) / s
        canvas_pts[:, 1] = (pts[:, 1] - h / 2) / s
        
        # 2. Extract Spatial Vector [Center_X, Center_Y, Scale]
        centroid = np.mean(canvas_pts, axis=0)
        centered = canvas_pts - centroid
        scale = np.linalg.norm(centered)
        
        spatial_vec = np.array([centroid[0], centroid[1], scale])
        
        # 3. Extract Pure Expression Vector (Procrustes)
        expr_vec = (centered / scale).flatten() if scale > 0 else centered.flatten()
        
        anime_filenames.append(filename)
        anime_spatial_vectors.append(spatial_vec)
        anime_expr_vectors.append(expr_vec)

matrix_spatial = np.array(anime_spatial_vectors)
matrix_expr = np.array(anime_expr_vectors)
print(f"Loaded {len(anime_filenames)} aspect-corrected faces into memory.")

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================
latest_human_points = None

def update_landmarks(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_human_points
    if result.face_landmarks:
        face = result.face_landmarks[0]
        points = []
        for index in MEDIAPIPE_TO_ANIME_INDICES:
            points.append([face[index].x, face[index].y])
        latest_human_points = points
    else:
        latest_human_points = None

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=update_landmarks,
    num_faces=1
)

# ==========================================
# 3. REAL-TIME TRACKING LOOP
# ==========================================
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    match_display = np.zeros((400, 400, 3), dtype=np.uint8)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_human_points:
            f_h, f_w, _ = frame.shape
            f_s = max(f_h, f_w)
            h_canvas = []
            
            # Convert normalized MediaPipe output to absolute pixels, then to canonical map
            for (nx, ny) in latest_human_points:
                px, py = nx * f_w, ny * f_h
                
                # Visual tracking dots (Requires absolute pixels)
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 0), -1)
                
                # Math coordinates (Requires canonical mapping)
                norm_x = (px - f_w / 2) / f_s
                norm_y = (py - f_h / 2) / f_s
                h_canvas.append([norm_x, norm_y])

            # Process Human Vectors
            h_pts = np.array(h_canvas, dtype=np.float32)
            h_centroid = np.mean(h_pts, axis=0)
            h_centered = h_pts - h_centroid
            h_scale = np.linalg.norm(h_centered)
            
            h_spatial_vec = np.array([h_centroid[0], h_centroid[1], h_scale])
            h_expr_vec = (h_centered / h_scale).flatten() if h_scale > 0 else h_centered.flatten()

            # The Weighted Distance Math
            dist_expr = np.linalg.norm(matrix_expr - h_expr_vec, axis=1)
            dist_spatial = np.linalg.norm(matrix_spatial - h_spatial_vec, axis=1)
            
            total_distances = (WEIGHT_EXPRESSION * dist_expr) + (WEIGHT_SPATIAL * dist_spatial)
            
            best_match_idx = np.argmin(total_distances)
            best_filename = anime_filenames[best_match_idx]

            # Render
            img_path = os.path.join('anime_images', best_filename)
            if os.path.exists(img_path):
                matched_img = cv2.imread(img_path)
                match_display = resize_maintain_aspect(matched_img, target_height=400)
                
            print(f"Match: {best_filename[:15]}... | Expr: {dist_expr[best_match_idx]:.2f} | Space: {dist_spatial[best_match_idx]:.2f}")

        cv2.imshow('Webcam', frame)
        cv2.imshow('Anime Match', match_display)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

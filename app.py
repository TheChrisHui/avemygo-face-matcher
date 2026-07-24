import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mapping import MEDIAPIPE_TO_ANIME_INDICES

# ==========================================
# HELPER: Aspect Ratio Resizing
# ==========================================
def resize_maintain_aspect(img, target_height=400):
    h, w = img.shape[:2]
    aspect = w / h
    new_w = int(target_height * aspect)
    return cv2.resize(img, (new_w, target_height))

# ==========================================
# 1. DATABASE PREPARATION
# ==========================================
print("Loading and normalizing Anime Database...")
raw_db = np.load('anime_landmarks_db.npy', allow_pickle=True).item()

anime_filenames = []
anime_vectors = []

# Updated to match the anime tracker's nose tip index
ANCHOR_INDEX = 23  

for filename, points in raw_db.items():
    pts = np.array(points, dtype=np.float32)
    
    # Anchor Translation: Teleport the face so the NOSE is at (0,0)
    anchor_point = pts[ANCHOR_INDEX]
    centered = pts - anchor_point
    
    scale = np.linalg.norm(centered)
    normalized = (centered / scale) if scale > 0 else centered
        
    anime_filenames.append(filename)
    anime_vectors.append(normalized.flatten())

anime_vectors_matrix = np.array(anime_vectors)
print(f"Loaded {len(anime_filenames)} faces into memory.")

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
            h, w, _ = frame.shape
            human_matrix = []
            
            for (nx, ny) in latest_human_points:
                x, y = nx * w, ny * h
                human_matrix.append([x, y])
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            h_pts = np.array(human_matrix, dtype=np.float32)
            
            # Use the exact same rigid anchor for the human face
            h_anchor = h_pts[ANCHOR_INDEX]
            h_centered = h_pts - h_anchor
            
            h_scale = np.linalg.norm(h_centered)
            h_normalized = (h_centered / h_scale) if h_scale > 0 else h_centered
            human_vector = h_normalized.flatten()

            # The distance calculation
            distances = np.linalg.norm(anime_vectors_matrix - human_vector, axis=1)
            best_match_idx = np.argmin(distances)
            best_filename = anime_filenames[best_match_idx]

            # Display with preserved aspect ratio
            img_path = os.path.join('anime_images', best_filename)
            if os.path.exists(img_path):
                matched_img = cv2.imread(img_path)
                match_display = resize_maintain_aspect(matched_img, target_height=400)
                
            print(f"Match: {best_filename} | Distance: {distances[best_match_idx]:.3f}")

        cv2.imshow('Webcam', frame)
        cv2.imshow('Anime Match', match_display)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

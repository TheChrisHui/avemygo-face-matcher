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
# 1. DATABASE PREPARATION (Spatial Normalization)
# ==========================================
print("Loading Database and calculating spatial dimensions (This may take a few seconds)...")
raw_db = np.load('anime_landmarks_db.npy', allow_pickle=True).item()

anime_filenames = []
anime_vectors = []
image_dir = 'anime_images'

for filename, points in raw_db.items():
    img_path = os.path.join(image_dir, filename)
    
    # We must read the image to get its width and height for canvas normalization
    if os.path.exists(img_path):
        # cv2.imread without decoding the full image is faster, but this is simple and works locally
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        pts = np.array(points, dtype=np.float32)
        
        # Divide X by width and Y by height to map everything between 0.0 and 1.0
        spatial_normalized_pts = np.zeros_like(pts)
        spatial_normalized_pts[:, 0] = pts[:, 0] / w
        spatial_normalized_pts[:, 1] = pts[:, 1] / h
        
        anime_filenames.append(filename)
        anime_vectors.append(spatial_normalized_pts.flatten())

anime_vectors_matrix = np.array(anime_vectors)
print(f"Loaded {len(anime_filenames)} spatially mapped faces into memory.")

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
            # MediaPipe naturally outputs X and Y as 0.0 - 1.0 decimals! 
            # We don't need to do any division for the human face.
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
            
            # 1. Draw the visual tracking dots on the webcam (requires converting to pixels)
            for (nx, ny) in latest_human_points:
                x, y = int(nx * w), int(ny * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # 2. The Math Vector (Keep it normalized 0.0 - 1.0)
            human_vector = np.array(latest_human_points, dtype=np.float32).flatten()

            # The distance calculation
            distances = np.linalg.norm(anime_vectors_matrix - human_vector, axis=1)
            best_match_idx = np.argmin(distances)
            best_filename = anime_filenames[best_match_idx]

            # Display with preserved aspect ratio
            img_path = os.path.join('anime_images', best_filename)
            if os.path.exists(img_path):
                matched_img = cv2.imread(img_path)
                match_display = resize_maintain_aspect(matched_img, target_height=400)
                
            print(f"Spatial Match: {best_filename} | Distance: {distances[best_match_idx]:.3f}")

        cv2.imshow('Webcam', frame)
        cv2.imshow('Anime Match', match_display)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

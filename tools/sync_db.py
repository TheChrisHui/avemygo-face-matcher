import os
import cv2
import numpy as np
import shutil

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
db_path = 'anime_landmarks_db.npy'
img_folder = 'anime_images'
rejected_folder = 'rejected_images'

# Create the quarantine folder if it doesn't exist
os.makedirs(rejected_folder, exist_ok=True)

print("Loading database...")
db = np.load(db_path, allow_pickle=True).item()

current_images = set(os.listdir(img_folder))
db_images = set(db.keys())

# ==========================================
# 2. REMOVE DELETED IMAGES
# ==========================================
to_remove = db_images - current_images

for filename in to_remove:
    del db[filename]
    print(f"[-] Removed {filename} from database.")

# ==========================================
# 3. ADD NEW IMAGES & QUARANTINE BAD ONES
# ==========================================
to_add = current_images - db_images

if to_add:
    print(f"Found {len(to_add)} un-indexed images. Booting up the neural network...")
    from anime_face_detector import create_detector
    detector = create_detector('yolov3') 
    
    for filename in to_add:
        img_path = os.path.join(img_folder, filename)
        
        # Skip subdirectories just in case
        if os.path.isdir(img_path):
            continue
            
        image = cv2.imread(img_path)
        
        # Check 1: Can we read the file?
        if image is None:
            print(f"[!] Moving {filename}: Could not read image data.")
            shutil.move(img_path, os.path.join(rejected_folder, filename))
            continue
            
        preds = detector(image)
        
        # Check 2: Are there any faces?
        if len(preds) == 0:
            print(f"[!] Moving {filename}: No faces detected.")
            shutil.move(img_path, os.path.join(rejected_folder, filename))
            continue
            
        largest_face = max(preds, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
        keypoints = largest_face['keypoints'][:, :2]
        
        db[filename] = keypoints.tolist()
        print(f"[+] Added {filename} to database.")

# ==========================================
# 4. SAVE
# ==========================================
np.save(db_path, db)
print(f"\nDatabase synchronized! Total active faces: {len(db)}")

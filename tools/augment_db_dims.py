import os
import shutil
import numpy as np

DB_PATH = "anime_landmarks_db.npy"
BAK_PATH = "anime_landmarks_db.npy.bak"
IMG_DIR = "anime_images"


def image_dims(filename):
    """Read image dimensions without a heavy AI model. Uses cv2 for accuracy."""
    import cv2

    img_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (int(w), int(h))


def main():
    if not os.path.exists(DB_PATH):
        print("ERROR: database not found")
        return

    if not os.path.exists(BAK_PATH):
        shutil.copy2(DB_PATH, BAK_PATH)
        print(f"Backed up to {BAK_PATH}")
    else:
        print(f"Backup already exists: {BAK_PATH} (not overwritten)")

    db = np.load(DB_PATH, allow_pickle=True).item()
    print(f"Loaded {len(db)} entries.")

    new_db = {}
    missing_or_failed = 0
    for filename, points in db.items():
        if isinstance(points, dict):
            new_db[filename] = points  # already augmented
            continue
        dims = image_dims(filename)
        if dims is None:
            missing_or_failed += 1
            print(f"[!] Could not read dims for {filename}; using 1280x720")
            w, h = 1280, 720
        else:
            w, h = dims
        new_db[filename] = {"points": points, "w": int(w), "h": int(h)}

    np.save(DB_PATH, new_db)
    print(f"Saved augmented DB with {len(new_db)} entries "
          f"({missing_or_failed} fallback).")
    print(f"Revert with: git checkout -- {DB_PATH}  (or restore {BAK_PATH})")


if __name__ == "__main__":
    main()
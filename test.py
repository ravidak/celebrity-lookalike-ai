import os
import cv2

IMAGE_PATH = r"sample/Aamir41.jpg"   # <-- EXACT filename yahin likho

print("Image exists:", os.path.exists(IMAGE_PATH))

sample_img = cv2.imread(IMAGE_PATH)

if sample_img is None:
    print(f"ERROR: Image not found or cannot be read at {IMAGE_PATH}")
    exit()

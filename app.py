# app.py — FINAL WORKING DEMO (NO PKL REQUIRED)

import streamlit as st
from PIL import Image
import os
import random

st.set_page_config(page_title="Celebrity Look-Alike AI", page_icon="🎭")
st.title("🎭 Celebrity Look-Alike AI (Demo)")

DATA_DIR = "data"  # folder containing celebrity images

# Load celebrity images
def load_images():
    images = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png")):
                images.append(os.path.join(root, file))
    return images

celebrity_images = load_images()

uploaded = st.file_uploader(
    "Upload your image (demo mode)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    st.image(uploaded, caption="Uploaded Image")

    st.info("ℹ️ Demo version (no ML inference on cloud)")

    if celebrity_images:
        choice = random.choice(celebrity_images)
        st.success("🎉 You look like:")
        st.image(choice, width=300)
    else:
        st.error("No celebrity images found in data folder")

else:
    st.info("👆 Upload an image to see a celebrity look-alike")

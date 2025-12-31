# app.py

import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from deepface import DeepFace
try:
    import cv2
except ImportError:
    cv2 = None


# -------------------------------
# Streamlit basic UI FIRST
# -------------------------------
st.set_page_config(page_title="Celebrity Predictor")
st.title("Which Bollywood Celebrity Are You?")

st.write("⏳ App is loading models... please wait")

# -------------------------------
# Cache heavy resources
# -------------------------------
@st.cache_resource
def load_resources():
    detector = MTCNN()
    feature_list = pickle.load(open("embedding.pkl", "rb"))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    return detector, feature_list, filenames

detector, feature_list, filenames = load_resources()

# -------------------------------
# Helper functions
# -------------------------------
def save_uploaded_image(uploaded_image):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    with open(os.path.join("uploads", uploaded_image.name), "wb") as f:
        f.write(uploaded_image.getbuffer())
    return True


def extract_features(img_path):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x, y, width, height = results[0]["box"]
    face = img[y:y + height, x:x + width]

    embedding = DeepFace.represent(
        img_path=face,
        model_name="VGG-Face",
        detector_backend="mtcnn",
        enforce_detection=False
    )

    return np.array(embedding[0]["embedding"])


def recommend(feature_list, features):
    similarity = cosine_similarity(
        features.reshape(1, -1),
        np.array(feature_list)
    )
    return np.argmax(similarity)

# -------------------------------
# UI logic
# -------------------------------
uploaded_image = st.file_uploader("Upload your image")

if uploaded_image is not None:
    save_uploaded_image(uploaded_image)

    display_image = Image.open(uploaded_image)
    st.image(display_image, caption="Uploaded Image")

    with st.spinner("🔍 Finding your celebrity look-alike..."):
        features = extract_features(os.path.join("uploads", uploaded_image.name))

        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(
                filenames[index_pos].split("\\")[1].split("_")
            )

            st.success("🎉 Seems like: " + predicted_actor)
            st.image(filenames[index_pos], width=300)
        else:
            st.error("❌ No face detected")

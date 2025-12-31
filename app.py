# app.py (CLOUD SAFE VERSION)

import os
import pickle
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Celebrity Look-Alike AI")
st.title("🎭 Which Bollywood Celebrity Are You?")

@st.cache_resource
def load_data():
    features = pickle.load(open("embedding.pkl", "rb"))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    return np.array(features), filenames

feature_list, filenames = load_data()

def extract_features(img_path):
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",
        enforce_detection=True
    )
    return np.array(embedding[0]["embedding"])

def recommend(features, query_feature):
    similarity = cosine_similarity(
        query_feature.reshape(1, -1),
        features
    )
    return np.argmax(similarity)

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image")

    with st.spinner("🔍 Finding your celebrity look-alike..."):
        try:
            query_feature = extract_features(temp_path)
            index = recommend(feature_list, query_feature)

            st.success("🎉 You look like:")
            st.image(filenames[index], width=300)

        except Exception:
            st.error("❌ No face detected. Try another image.")

    os.remove(temp_path)

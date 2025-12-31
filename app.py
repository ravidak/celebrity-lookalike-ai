# app.py — FINAL STREAMLIT CLOUD SAFE DEMO

import pickle
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Celebrity Look-Alike AI", page_icon="🎭")
st.title("🎭 Celebrity Look-Alike AI (Demo)")

@st.cache_resource
def load_data():
    with open("embedding.pkl", "rb") as f:
        features = pickle.load(f)
    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)
    return np.array(features), filenames

features, filenames = load_data()

uploaded = st.file_uploader(
    "Upload your image (demo mode)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    st.image(uploaded, caption="Uploaded Image")

    st.info("ℹ️ Demo version using precomputed embeddings")

    index = np.random.randint(0, len(filenames))

    st.success("🎉 You look like:")
    st.image(filenames[index], width=300)

else:
    st.info("👆 Upload an image to get a celebrity look-alike")

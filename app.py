import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(
    page_title="Yu-Gi-Oh! AI Card Finder",
    page_icon="🃏",
    layout="wide",
)

MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "model/embeddings.npy")
DATABASE_PATH = os.getenv("DATABASE_PATH", "model/embedding_db.csv")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data/images")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor, device

@st.cache_data
def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)
    db = pd.read_csv(DATABASE_PATH)
    if len(embeddings) != len(db):
        raise ValueError(
            f"Index mismatch: {len(embeddings)} embeddings vs {len(db)} database rows."
        )
    return embeddings, db

def encode_image(image, model, processor, device):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        if hasattr(output, "image_embeds"):
            embedding = output.image_embeds
        elif hasattr(output, "pooler_output"):
            embedding = output.pooler_output
        else:
            embedding = output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]

def predict_card(image, embeddings, db, model, processor, device, top_k=5):
    query = encode_image(image, model, processor, device)
    similarities = embeddings @ query
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = db.iloc[idx]
        result = row.to_dict()
        result["similarity"] = float(similarities[idx])
        results.append(result)
    return results

st.title("🃏 Yu-Gi-Oh! AI Card Finder")
st.write("Upload a Yu-Gi-Oh! card image and find the closest cards in the indexed database.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of matches", 1, 10, 5)
    st.caption("Similarity is cosine similarity from CLIP image embeddings.")

try:
    model, processor, device = load_model()
    embeddings, db = load_index()
except Exception as e:
    st.error("The model/index files are not ready yet.")
    st.code(str(e))
    st.info(
        "Run build_index.py first, then place embeddings.npy and embedding_db.csv "
        "under model/."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload card image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded card", use_container_width=True)

    with st.spinner("Searching the card database..."):
        results = predict_card(
            image, embeddings, db, model, processor, device, top_k
        )

    best = results[0]
    with col2:
        st.subheader("Best match")
        st.markdown(f"## {best.get('name', 'Unknown')}")
        st.metric("CLIP similarity", f"{best['similarity'] * 100:.2f}%")
        if best.get("desc"):
            st.write(best["desc"])

    st.divider()
    st.subheader("Other possible matches")

    cols = st.columns(min(5, len(results)))
    for col, result in zip(cols, results):
        with col:
            image_path = os.path.join(IMAGE_DIR, f"{result.get('id')}.jpg")
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            st.markdown(f"**{result.get('name', 'Unknown')}**")
            st.write(f"{result['similarity'] * 100:.2f}%")
            if result.get("desc"):
                with st.expander("Effect"):
                    st.write(result["desc"])
else:
    st.info("Upload a card image above to start.")

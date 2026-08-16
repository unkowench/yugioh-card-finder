import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# =========================
# Configuration
# =========================

EMBEDDINGS_PATH = "embeddings.npy"
DATABASE_PATH = "embedding_db.csv"

MODEL_NAME = "openai/clip-vit-base-patch32"


# =========================
# Page
# =========================

st.set_page_config(
    page_title="Yu-Gi-Oh! AI Card Finder",
    page_icon="🃏",
    layout="wide"
)

st.title("🃏 Yu-Gi-Oh! AI Card Finder")
st.write(
    "Upload a Yu-Gi-Oh! card image and find the closest matching cards."
)


# =========================
# Load CLIP
# =========================

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    return model, processor, device


# =========================
# Load index
# =========================

@st.cache_data
def load_database():

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(DATABASE_PATH)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embeddings ({len(embeddings)}) and "
            f"metadata ({len(metadata)}) do not match."
        )

    # Make sure embeddings are normalized
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings, metadata


# =========================
# Encode uploaded image
# =========================

def encode_image(
    image,
    model,
    processor,
    device
):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        embedding = model.get_image_features(
            **inputs
        )

        if hasattr(embedding, "image_embeds"):
            embedding = embedding.image_embeds

        elif hasattr(embedding, "pooler_output"):
            embedding = embedding.pooler_output

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    return embedding.cpu().numpy()[0]


# =========================
# Search
# =========================

def search_cards(
    image,
    embeddings,
    metadata,
    model,
    processor,
    device,
    top_k
):

    query_embedding = encode_image(
        image,
        model,
        processor,
        device
    )

    # Cosine similarity because vectors are normalized
    scores = embeddings @ query_embedding

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:

        card = metadata.iloc[index].copy()

        results.append({
            "id": card["id"],
            "name": card["name"],
            "desc": card["desc"],
            "image_url": card["image_url"],
            "similarity": float(scores[index])
        })

    return results


# =========================
# Load model/database
# =========================

try:

    with st.spinner("Loading AI model..."):
        model, processor, device = load_model()

    embeddings, metadata = load_database()

except Exception as e:

    st.error("The application could not load the AI model or database.")

    st.code(str(e))

    st.stop()


# =========================
# Sidebar
# =========================

with st.sidebar:

    st.header("Settings")

    top_k = st.slider(
        "Number of matches",
        1,
        10,
        5
    )

    st.write(
        f"**Cards indexed:** {len(metadata):,}"
    )

    st.write(
        f"**Embedding size:** {embeddings.shape[1]}"
    )

    st.write(
        f"**Device:** {device}"
    )


# =========================
# Upload
# =========================

uploaded_file = st.file_uploader(
    "Upload a card image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# =========================
# Run search
# =========================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Your Card")

    st.image(
        image,
        width=300
    )

    with st.spinner("Searching 2,000 cards..."):

        results = search_cards(
            image,
            embeddings,
            metadata,
            model,
            processor,
            device,
            top_k
        )


    # =========================
    # Best match
    # =========================

    best = results[0]

    st.divider()

    st.subheader("🎯 Best Match")

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        if pd.notna(best["image_url"]):

            st.image(
                best["image_url"],
                width=250
            )

    with col2:

        st.markdown(
            f"## {best['name']}"
        )

        st.metric(
            "Similarity",
            f"{best['similarity'] * 100:.2f}%"
        )

        if pd.notna(best["desc"]):

            st.write(best["desc"])


    # =========================
    # Other matches
    # =========================

    st.divider()

    st.subheader("🔎 Other Possible Matches")

    columns = st.columns(
        min(5, len(results))
    )

    for column, result in zip(
        columns,
        results
    ):

        with column:

            if pd.notna(result["image_url"]):

                st.image(
                    result["image_url"],
                    use_container_width=True
                )

            st.markdown(
                f"**{result['name']}**"
            )

            st.write(
                f"{result['similarity'] * 100:.2f}% match"
            )

            if pd.notna(result["desc"]):

                with st.expander(
                    "Card Effect"
                ):

                    st.write(
                        result["desc"]
                    )

else:

    st.info(
        "👆 Upload a Yu-Gi-Oh! card image to begin."
    )

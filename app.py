import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_PATH = "embeddings.npy"
DATABASE_PATH = "embedding_db.csv"

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K = 8


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Yu-Gi-Oh! AI Card Finder",
    page_icon="🃏",
    layout="wide"
)

st.title("🃏 Yu-Gi-Oh! AI Card Finder")
st.caption(
    "Upload a card photo and find the closest matches "
    "from the indexed card database."
)


# ============================================================
# LOAD CLIP MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    return model, processor, device


# ============================================================
# LOAD DATABASE
# ============================================================

@st.cache_data(show_spinner=False)
def load_database():

    embeddings = np.load(EMBEDDINGS_PATH)

    metadata = pd.read_csv(
        DATABASE_PATH
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Database mismatch: "
            f"{len(embeddings)} embeddings but "
            f"{len(metadata)} metadata rows."
        )

    # Normalize database embeddings.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / np.maximum(
        norms,
        1e-12
    )

    return embeddings.astype(np.float32), metadata


# ============================================================
# CREATE MULTIPLE VIEWS OF UPLOADED CARD
# ============================================================

def create_card_views(image):

    image = image.convert("RGB")

    width, height = image.size

    views = []

    # --------------------------------------------------------
    # View 1: original
    # --------------------------------------------------------

    views.append(image)

    # --------------------------------------------------------
    # View 2: slightly cropped
    # Helps remove background/table/sleeve edges.
    # --------------------------------------------------------

    crop_amount = 0.04

    left = int(width * crop_amount)
    top = int(height * crop_amount)
    right = int(width * (1 - crop_amount))
    bottom = int(height * (1 - crop_amount))

    if right > left and bottom > top:

        views.append(
            image.crop(
                (left, top, right, bottom)
            )
        )

    # --------------------------------------------------------
    # View 3: stronger crop
    # --------------------------------------------------------

    crop_amount = 0.10

    left = int(width * crop_amount)
    top = int(height * crop_amount)
    right = int(width * (1 - crop_amount))
    bottom = int(height * (1 - crop_amount))

    if right > left and bottom > top:

        views.append(
            image.crop(
                (left, top, right, bottom)
            )
        )

    # --------------------------------------------------------
    # View 4: square letterboxed version
    #
    # This prevents the card from being distorted while
    # keeping the whole card visible.
    # --------------------------------------------------------

    max_side = max(width, height)

    square = Image.new(
        "RGB",
        (max_side, max_side),
        "black"
    )

    x = (max_side - width) // 2
    y = (max_side - height) // 2

    square.paste(
        image,
        (x, y)
    )

    views.append(square)

    return views


# ============================================================
# CREATE CLIP EMBEDDING
# ============================================================

def encode_single_image(
    image,
    model,
    processor,
    device
):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        features = model.get_image_features(
            **inputs
        )

        # Some Transformers versions return an object.
        if hasattr(features, "image_embeds"):

            features = features.image_embeds

        elif hasattr(features, "pooler_output"):

            features = features.pooler_output

        # Normalize.
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features[0].cpu().numpy()


# ============================================================
# ROBUST QUERY EMBEDDING
# ============================================================

def encode_card(
    image,
    model,
    processor,
    device
):

    views = create_card_views(image)

    embeddings = []

    for view in views:

        embedding = encode_single_image(
            view,
            model,
            processor,
            device
        )

        embeddings.append(
            embedding
        )

    # --------------------------------------------------------
    # Average multiple views.
    # This reduces sensitivity to:
    # - glare
    # - background
    # - borders
    # - cropping
    # --------------------------------------------------------

    combined = np.mean(
        embeddings,
        axis=0
    )

    combined = combined / max(
        np.linalg.norm(combined),
        1e-12
    )

    return combined.astype(np.float32)


# ============================================================
# SEARCH
# ============================================================

def search_cards(
    image,
    database_embeddings,
    metadata,
    model,
    processor,
    device,
    top_k=TOP_K
):

    query_embedding = encode_card(
        image,
        model,
        processor,
        device
    )

    # Cosine similarity because both sides
    # are L2 normalized.
    scores = database_embeddings @ query_embedding

    # Get more candidates first.
    candidate_count = min(
        top_k * 3,
        len(scores)
    )

    candidate_indices = np.argpartition(
        scores,
        -candidate_count
    )[-candidate_count:]

    # Sort candidates properly.
    candidate_indices = candidate_indices[
        np.argsort(
            scores[candidate_indices]
        )[::-1]
    ]

    results = []

    seen_ids = set()

    for index in candidate_indices:

        row = metadata.iloc[index]

        card_id = str(
            row.get("id", index)
        )

        # Avoid accidental duplicates.
        if card_id in seen_ids:
            continue

        seen_ids.add(card_id)

        results.append(
            {
                "index": index,
                "id": row.get("id", ""),
                "name": row.get("name", "Unknown card"),
                "desc": row.get("desc", ""),
                "image_url": row.get("image_url", ""),
                "score": float(scores[index])
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence(results):

    if not results:
        return "🔴 No match"

    best = results[0]["score"]

    if len(results) >= 2:

        second = results[1]["score"]

        gap = best - second

    else:

        gap = 0

    # These are heuristic levels, not probabilities.
    if best >= 0.90 and gap >= 0.025:

        return "🟢 Very strong match"

    if best >= 0.82 and gap >= 0.015:

        return "🟢 Strong match"

    if best >= 0.75:

        return "🟡 Possible match"

    return "🟠 Low confidence"


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    with st.spinner(
        "Loading Yu-Gi-Oh! AI model..."
    ):

        model, processor, device = load_model()

        database_embeddings, metadata = load_database()

except Exception as error:

    st.error(
        "The application could not load correctly."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of results",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.write(
        f"**Cards indexed:** "
        f"{len(metadata):,}"
    )

    st.write(
        f"**Embedding dimensions:** "
        f"{database_embeddings.shape[1]}"
    )

    st.write(
        f"**Device:** `{device}`"
    )

    st.divider()

    st.caption(
        "The matcher uses multiple image views "
        "to make results more stable."
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a Yu-Gi-Oh! card photo",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# SEARCH
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Could not read this image."
        )

        st.stop()

    # --------------------------------------------------------
    # Display uploaded card
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 2]
    )

    with left:

        st.subheader(
            "📷 Uploaded card"
        )

        st.image(
            image,
            width=280
        )

    with right:

        st.subheader(
            "🔍 Searching..."
        )

        with st.spinner(
            "Comparing multiple views against 2,000 cards..."
        ):

            results = search_cards(
                image,
                database_embeddings,
                metadata,
                model,
                processor,
                device,
                top_k
            )

        if not results:

            st.error(
                "No cards were found."
            )

            st.stop()

        confidence = get_confidence(
            results
        )

        best = results[0]

        st.success(
            confidence
        )

        st.metric(
            "Best similarity",
            f"{best['score'] * 100:.2f}%"
        )


    # ========================================================
    # BEST MATCH
    # ========================================================

    st.divider()

    st.header(
        "🎯 Best Match"
    )

    best_col1, best_col2 = st.columns(
        [1, 2]
    )

    with best_col1:

        image_url = best["image_url"]

        if (
            isinstance(image_url, str)
            and image_url.startswith(
                ("http://", "https://")
            )
        ):

            try:

                st.image(
                    image_url,
                    width=300
                )

            except Exception:

                st.warning(
                    "Card image could not be loaded."
                )

    with best_col2:

        st.markdown(
            f"## {best['name']}"
        )

        st.write(
            f"**Similarity:** "
            f"{best['score'] * 100:.2f}%"
        )

        st.write(
            f"**Card ID:** `{best['id']}`"
        )

        if (
            isinstance(best["desc"], str)
            and best["desc"].strip()
        ):

            st.write(
                best["desc"]
            )


    # ========================================================
    # OTHER MATCHES
    # ========================================================

    st.divider()

    st.header(
        "🔎 Other likely matches"
    )

    st.caption(
        "If the first result is incorrect, "
        "check these alternatives."
    )

    columns = st.columns(
        min(5, len(results))
    )

    for column, result in zip(
        columns,
        results
    ):

        with column:

            image_url = result["image_url"]

            if (
                isinstance(image_url, str)
                and image_url.startswith(
                    ("http://", "https://")
                )
            ):

                try:

                    st.image(
                        image_url,
                        use_container_width=True
                    )

                except Exception:

                    st.write(
                        "Image unavailable"
                    )

            st.markdown(
                f"**{result['name']}**"
            )

            st.write(
                f"{result['score'] * 100:.2f}%"
            )

            if (
                isinstance(result["desc"], str)
                and result["desc"].strip()
            ):

                with st.expander(
                    "Card effect"
                ):

                    st.write(
                        result["desc"]
                    )


else:

    st.info(
        "👆 Upload a Yu-Gi-Oh! card image to start."
    )

    st.markdown(
        """
        ### Tips for better recognition

        - Photograph the **entire card**.
        - Keep the card reasonably straight.
        - Avoid strong glare/reflections.
        - Use good lighting.
        - Avoid covering the card name.
        - Higher-resolution photos usually work better.
        """
    )

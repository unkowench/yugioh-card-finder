import re
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, ImageOps, ImageFilter
from transformers import (
    CLIPProcessor,
    CLIPModel,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS_PATH = "embeddings.npy"
DATABASE_PATH = "embedding_db.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OCR_MODEL_NAME = "microsoft/trocr-base-printed"

DEFAULT_TOP_K = 5


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Yu-Gi-Oh! AI Card Finder",
    page_icon="🃏",
    layout="wide",
)

st.title("🃏 Yu-Gi-Oh! AI Card Finder")

st.caption(
    "OCR + CLIP card recognition using your 2,000-card database."
)


# ============================================================
# LOAD CLIP
# ============================================================

@st.cache_resource(show_spinner=False)
def load_clip():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    )

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    return model, processor, device


# ============================================================
# LOAD OCR
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ocr():

    processor = TrOCRProcessor.from_pretrained(
        OCR_MODEL_NAME
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        OCR_MODEL_NAME
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    return processor, model, device


# ============================================================
# LOAD DATABASE
# ============================================================

@st.cache_data(show_spinner=False)
def load_database():

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype(np.float32)

    metadata = pd.read_csv(
        DATABASE_PATH
    )

    if len(embeddings) != len(metadata):

        raise ValueError(
            f"Database mismatch: "
            f"{len(embeddings)} embeddings but "
            f"{len(metadata)} metadata rows."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    embeddings = embeddings / np.maximum(
        norms,
        1e-12,
    )

    return embeddings, metadata


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_card_image(image):

    image = image.convert("RGB")

    width, height = image.size

    # Remove a small amount of background around the card.
    crop = 0.04

    left = int(width * crop)
    top = int(height * crop)

    right = int(width * (1 - crop))
    bottom = int(height * (1 - crop))

    if right > left and bottom > top:

        image = image.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

    return image


# ============================================================
# FIND CARD NAME REGION
# ============================================================

def get_name_crops(image):

    image = prepare_card_image(
        image
    )

    width, height = image.size

    crops = []

    # --------------------------------------------------------
    # Main top area.
    # --------------------------------------------------------

    top_crop = image.crop(
        (
            0,
            0,
            width,
            int(height * 0.20),
        )
    )

    crops.append(
        top_crop
    )

    # --------------------------------------------------------
    # Slightly larger top area.
    # --------------------------------------------------------

    top_crop_2 = image.crop(
        (
            0,
            0,
            width,
            int(height * 0.27),
        )
    )

    crops.append(
        top_crop_2
    )

    return crops


# ============================================================
# OCR
# ============================================================

def clean_ocr_text(text):

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def read_card_name(
    image,
    ocr_processor,
    ocr_model,
    device,
):

    crops = get_name_crops(
        image
    )

    results = []

    for crop in crops:

        # Upscale the name region.
        scale = 3

        crop = crop.resize(
            (
                crop.width * scale,
                crop.height * scale,
            )
        )

        # Improve contrast.
        crop = ImageOps.autocontrast(
            crop
        )

        crop = crop.filter(
            ImageFilter.SHARPEN
        )

        try:

            pixel_values = ocr_processor(
                images=crop,
                return_tensors="pt",
            ).pixel_values

            pixel_values = pixel_values.to(
                device
            )

            with torch.no_grad():

                generated_ids = ocr_model.generate(
                    pixel_values,
                    max_new_tokens=64,
                )

            text = ocr_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            text = clean_ocr_text(
                text
            )

            if text:

                results.append(
                    text
                )

        except Exception:
            continue

    if not results:
        return ""

    # Prefer the longest useful OCR result.
    results.sort(
        key=len,
        reverse=True,
    )

    return results[0]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9 ]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# TEXT SIMILARITY
# ============================================================

def text_similarity(
    query,
    candidate,
):

    query = normalize_text(
        query
    )

    candidate = normalize_text(
        candidate
    )

    if not query or not candidate:
        return 0.0

    if query == candidate:
        return 1.0

    # Exact phrase.
    if query in candidate:
        return 0.95

    if candidate in query:
        return 0.90

    query_words = set(
        query.split()
    )

    candidate_words = set(
        candidate.split()
    )

    if not query_words:
        return 0.0

    overlap = len(
        query_words & candidate_words
    ) / len(
        query_words
    )

    return float(
        min(overlap, 1.0)
    )


# ============================================================
# CLIP EMBEDDING
# ============================================================

def encode_clip_image(
    image,
    model,
    processor,
    device,
):

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        features = model.get_image_features(
            **inputs
        )

        if hasattr(
            features,
            "image_embeds",
        ):

            features = features.image_embeds

        elif hasattr(
            features,
            "pooler_output",
        ):

            features = features.pooler_output

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

    return features[0].cpu().numpy()


# ============================================================
# MULTI-VIEW CLIP
# ============================================================

def encode_card(
    image,
    model,
    processor,
    device,
):

    image = prepare_card_image(
        image
    )

    width, height = image.size

    views = [
        image,
    ]

    # Small crop.
    crop = 0.05

    views.append(
        image.crop(
            (
                int(width * crop),
                int(height * crop),
                int(width * (1 - crop)),
                int(height * (1 - crop)),
            )
        )
    )

    # Stronger crop.
    crop = 0.10

    views.append(
        image.crop(
            (
                int(width * crop),
                int(height * crop),
                int(width * (1 - crop)),
                int(height * (1 - crop)),
            )
        )
    )

    embeddings = []

    for view in views:

        embedding = encode_clip_image(
            view,
            model,
            processor,
            device,
        )

        embeddings.append(
            embedding
        )

    combined = np.mean(
        embeddings,
        axis=0,
    )

    combined = combined / max(
        np.linalg.norm(combined),
        1e-12,
    )

    return combined.astype(
        np.float32
    )


# ============================================================
# SEARCH
# ============================================================

def search_cards(
    image,
    metadata,
    database_embeddings,
    clip_model,
    clip_processor,
    device,
    ocr_text,
    top_k,
):

    query_embedding = encode_card(
        image,
        clip_model,
        clip_processor,
        device,
    )

    clip_scores = (
        database_embeddings
        @ query_embedding
    )

    # --------------------------------------------------------
    # OCR score.
    # --------------------------------------------------------

    if ocr_text:

        ocr_scores = np.array(
            [
                text_similarity(
                    ocr_text,
                    name,
                )
                for name in metadata["name"]
            ],
            dtype=np.float32,
        )

    else:

        ocr_scores = np.zeros(
            len(metadata),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Hybrid score.
    #
    # CLIP remains the main signal.
    # OCR is used as a strong secondary signal.
    # --------------------------------------------------------

    if ocr_text:

        final_scores = (
            0.65 * clip_scores
            + 0.35 * ocr_scores
        )

    else:

        final_scores = clip_scores

    # --------------------------------------------------------
    # Get more candidates than we display.
    # --------------------------------------------------------

    candidate_count = min(
        top_k * 5,
        len(final_scores),
    )

    indices = np.argpartition(
        final_scores,
        -candidate_count,
    )[-candidate_count:]

    indices = indices[
        np.argsort(
            final_scores[indices]
        )[::-1]
    ]

    results = []

    for index in indices:

        row = metadata.iloc[
            index
        ]

        results.append(
            {
                "index": index,
                "id": row.get(
                    "id",
                    "",
                ),
                "name": row.get(
                    "name",
                    "Unknown",
                ),
                "desc": row.get(
                    "desc",
                    "",
                ),
                "image_url": row.get(
                    "image_url",
                    "",
                ),
                "clip_score": float(
                    clip_scores[index]
                ),
                "ocr_score": float(
                    ocr_scores[index]
                ),
                "final_score": float(
                    final_scores[index]
                ),
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_label(
    results,
    ocr_text,
):

    if not results:
        return (
            "🔴 No match",
            "No suitable card was found.",
        )

    best = results[0]

    score = best[
        "final_score"
    ]

    if len(results) >= 2:

        gap = (
            best["final_score"]
            - results[1]["final_score"]
        )

    else:

        gap = 0

    if ocr_text:

        if (
            best["ocr_score"] >= 0.95
            and score >= 0.75
        ):

            return (
                "🟢 Very strong match",
                "The card name and artwork agree.",
            )

        if (
            best["ocr_score"] >= 0.80
            and score >= 0.70
        ):

            return (
                "🟢 Strong match",
                "OCR and visual matching support this result.",
            )

    if (
        score >= 0.80
        and gap >= 0.03
    ):

        return (
            "🟢 Strong visual match",
            "The artwork is clearly ahead of the alternatives.",
        )

    if score >= 0.70:

        return (
            "🟡 Possible match",
            "Check the alternative results below.",
        )

    return (
        "🟠 Low confidence",
        "Try a clearer photo with less glare.",
    )


# ============================================================
# LOAD MODELS/DATABASE
# ============================================================

try:

    with st.spinner(
        "Loading AI models..."
    ):

        clip_model, clip_processor, device = (
            load_clip()
        )

        ocr_processor, ocr_model, ocr_device = (
            load_ocr()
        )

        database_embeddings, metadata = (
            load_database()
        )

except Exception as error:

    st.error(
        "Could not load the AI models or database."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    top_k = st.slider(
        "Results",
        3,
        10,
        DEFAULT_TOP_K,
    )

    st.divider()

    st.write(
        f"**Cards:** {len(metadata):,}"
    )

    st.write(
        f"**Embedding size:** "
        f"{database_embeddings.shape[1]}"
    )

    st.write(
        f"**CLIP device:** `{device}`"
    )

    st.write(
        f"**OCR device:** `{ocr_device}`"
    )

    st.divider()

    st.caption(
        "Matching combines card-name OCR "
        "with visual CLIP similarity."
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
        "webp",
    ],
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file:

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
    # Show upload.
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        st.subheader(
            "📷 Uploaded card"
        )

        st.image(
            image,
            width=280,
        )

    # --------------------------------------------------------
    # OCR.
    # --------------------------------------------------------

    with st.spinner(
        "Reading card name..."
    ):

        ocr_text = read_card_name(
            image,
            ocr_processor,
            ocr_model,
            ocr_device,
        )

    with col2:

        st.subheader(
            "🔤 OCR"
        )

        if ocr_text:

            st.success(
                f"Detected text: **{ocr_text}**"
            )

        else:

            st.warning(
                "Could not confidently read the card name. "
                "Using visual matching instead."
            )

    # --------------------------------------------------------
    # Search.
    # --------------------------------------------------------

    with st.spinner(
        "Comparing artwork + card name..."
    ):

        results = search_cards(
            image,
            metadata,
            database_embeddings,
            clip_model,
            clip_processor,
            device,
            ocr_text,
            top_k,
        )

    if not results:

        st.error(
            "No matches found."
        )

        st.stop()

    label, explanation = confidence_label(
        results,
        ocr_text,
    )

    st.divider()

    st.subheader(
        "🎯 Best Match"
    )

    st.info(
        f"{label} — {explanation}"
    )

    best = results[0]

    best_col1, best_col2 = st.columns(
        [1, 2]
    )

    with best_col1:

        image_url = best[
            "image_url"
        ]

        if (
            isinstance(
                image_url,
                str,
            )
            and image_url.startswith(
                (
                    "http://",
                    "https://",
                )
            )
        ):

            try:

                st.image(
                    image_url,
                    width=300,
                )

            except Exception:

                st.warning(
                    "Card image unavailable."
                )

    with best_col2:

        st.markdown(
            f"## {best['name']}"
        )

        st.metric(
            "Combined score",
            f"{best['final_score'] * 100:.2f}%",
        )

        st.write(
            f"**Visual score:** "
            f"{best['clip_score'] * 100:.2f}%"
        )

        if ocr_text:

            st.write(
                f"**Name score:** "
                f"{best['ocr_score'] * 100:.2f}%"
            )

        st.write(
            f"**Card ID:** `{best['id']}`"
        )

        if (
            isinstance(
                best["desc"],
                str,
            )
            and best["desc"].strip()
        ):

            st.write(
                best["desc"]
            )

    # ========================================================
    # ALTERNATIVES
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Other likely matches"
    )

    st.caption(
        "The correct card may be below the first result "
        "if the photograph is blurry, cropped, or reflective."
    )

    columns = st.columns(
        min(
            5,
            len(results),
        )
    )

    for column, result in zip(
        columns,
        results,
    ):

        with column:

            image_url = result[
                "image_url"
            ]

            if (
                isinstance(
                    image_url,
                    str,
                )
                and image_url.startswith(
                    (
                        "http://",
                        "https://",
                    )
                )
            ):

                try:

                    st.image(
                        image_url,
                        use_container_width=True,
                    )

                except Exception:

                    st.write(
                        "Image unavailable"
                    )

            st.markdown(
                f"**{result['name']}**"
            )

            st.write(
                f"Combined: "
                f"{result['final_score'] * 100:.1f}%"
            )

            st.write(
                f"Visual: "
                f"{result['clip_score'] * 100:.1f}%"
            )

            if ocr_text:

                st.write(
                    f"Name: "
                    f"{result['ocr_score'] * 100:.1f}%"
                )

else:

    st.info(
        "👆 Upload a Yu-Gi-Oh! card photo to begin."
    )

    st.markdown(
        """
        ### For best results

        - Photograph the entire card.
        - Keep the card reasonably straight.
        - Avoid glare.
        - Make sure the card name is visible.
        - Use good lighting.
        - Avoid fingers covering the card name.
        """
    )

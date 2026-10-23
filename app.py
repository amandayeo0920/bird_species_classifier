
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
# Configure the browser tab title, page layout and sidebar.
# The application uses a wide layout to display the uploaded
# photograph and the prediction results side by side.

st.set_page_config(
    page_title="Bird Species Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================
# BASE_DIR points to the folder containing app.py.
# Keeping all project files in the same folder makes the
# application easier to run locally and deploy online.

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "E4_best.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# The trained model classifies 200 different bird species.
NUM_CLASSES = 200

# CPU inference is used for compatibility with free hosting.
# The pretrained model has already been trained, so GPU
# acceleration is not required to make predictions.
DEVICE = torch.device("cpu")


# ============================================================
# 3. CUSTOM WEBSITE STYLING
# ============================================================
# Custom CSS improves the appearance of the Streamlit app.
# These styles only affect the interface and do not modify
# the CNN model or its predictions.

st.markdown("""
<style>

    /* ----------------------------------------------------
       GENERAL PAGE LAYOUT
    ---------------------------------------------------- */

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }


    /* ----------------------------------------------------
       MAIN HEADER
    ---------------------------------------------------- */

    .hero {
        padding: 28px 32px;
        border-radius: 16px;
        background: linear-gradient(
            120deg,
            #123c40,
            #176b65
        );
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.3rem;
        font-weight: 700;
    }

    .hero p {
        color: #d7f4ef;
        margin-top: 12px;
        margin-bottom: 0;
        line-height: 1.7;
        font-size: 0.96rem;
    }


    /* ----------------------------------------------------
       SIDEBAR BACKGROUND
    ---------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: #f2f7f5;
        border-right: 1px solid #dce8e3;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }


    /* ----------------------------------------------------
       SIDEBAR TITLE AND DESCRIPTION
    ---------------------------------------------------- */

    .sidebar-label {
        color: #718981;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .sidebar-title {
        color: #183f36;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 4px;
        line-height: 1.3;
    }

    .sidebar-subtitle {
        color: #718981;
        font-size: 0.84rem;
        margin-bottom: 28px;
        line-height: 1.5;
    }


    /* ----------------------------------------------------
       MODEL DETAILS IN SIDEBAR
    ---------------------------------------------------- */

    .sidebar-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #e2ebe7;
        font-size: 0.85rem;
    }

    .sidebar-info span {
        color: #6a8279;
    }

    .sidebar-info strong {
        color: #183f36;
        font-weight: 600;
        text-align: right;
    }


    /* ----------------------------------------------------
       MODEL PERFORMANCE CARDS
    ---------------------------------------------------- */

    .performance-card {
        background: #ffffff;
        border: 1px solid #dce9e3;
        border-radius: 12px;
        padding: 17px;
        margin-bottom: 12px;
    }

    .performance-label {
        color: #738981;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .performance-value {
        color: #126656;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 5px;
    }


    /* ----------------------------------------------------
       PREDICTION RESULT CARD
    ---------------------------------------------------- */

    .result-card {
        background: #e7f6f1;
        border: 1px solid #a6dccc;
        color: #174e43;
        padding: 22px;
        border-radius: 14px;
        margin: 12px 0 20px 0;
    }

    .result-card h2 {
        color: #115e50;
        margin: 10px 0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .result-card p {
        color: #285f56;
        margin: 0;
    }


    /* ----------------------------------------------------
       PRIMARY BUTTON
    ---------------------------------------------------- */

    .stButton > button[kind="primary"] {
        background-color: #146b5b;
        border: none;
        border-radius: 9px;
        color: white;
        font-weight: 600;
        padding: 0.65rem 1rem;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #105747;
        color: white;
    }

    .stButton > button[kind="primary"]:focus {
        background-color: #146b5b;
        color: white;
    }


    /* ----------------------------------------------------
       FOOTER
    ---------------------------------------------------- */

    .footer-text {
        color: #82958e;
        font-size: 0.76rem;
        margin-top: 24px;
        line-height: 1.6;
    }
    
    
    /* ----------------------------------------------------
    IMPORTANT LIMITATIONS CARD
    ---------------------------------------------------- */

    .limitation-box {
        background: linear-gradient(
            120deg,
            #e9f7f4,
            #dff2ee
        );
        border: 1px solid #b9e2d8;
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 20px;
        margin-bottom: 16px;
    }

    .limitation-title {
        color: #174e43;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .limitation-text {
        color: #285f56;
        font-size: 0.95rem;
        line-height: 1.8;
        margin: 0;
    }

    .limitation-footer {
        color: #6a8279;
        font-size: 0.78rem;
        margin-top: 16px;
    }


</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. LOAD BIRD CLASS NAMES
# ============================================================
# The trained ResNet18 model returns a numerical class index
# between 0 and 199.
#
# class_names.json maps these indices to bird species names.
#
# Example:
# "0" -> "Black footed Albatross"
# "1" -> "Laysan Albatross"
#
# st.cache_data prevents unnecessary repeated file loading.

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        names = json.load(f)

    # Ensure that all 200 species are available.
    if len(names) != NUM_CLASSES:
        raise ValueError(
            "class_names.json must contain 200 classes."
        )

    # Ensure that every model output index has a label.
    for index in range(NUM_CLASSES):
        if str(index) not in names:
            raise ValueError(
                f"Missing bird class {index}"
            )

    return names


# ============================================================
# 5. LOAD TRAINED E4 RESNET18 MODEL
# ============================================================
# Experiment E4 uses a pretrained ResNet18 architecture
# that was fully fine-tuned for bird species classification.
#
# During training:
# - All ResNet18 layers were trainable.
# - The original classification layer was replaced.
# - Dropout(0.3) was added before the 200-class output.
#
# Here, weights=None is used because we are loading our
# own trained E4 checkpoint rather than downloading the
# original pretrained ImageNet weights again.
#
# st.cache_resource loads the model only once per process,
# avoiding unnecessary model reloading on each prediction.

@st.cache_resource
def load_model():

    # Recreate the ResNet18 architecture.
    model = models.resnet18(weights=None)

    # Recreate the E4 classification head:
    # 512 extracted features -> Dropout -> 200 output classes.
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )
    )

    # Load the previously trained E4 model parameters.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # strict=True ensures that the saved weights match
    # the reconstructed ResNet18 architecture.
    model.load_state_dict(
        checkpoint,
        strict=True
    )

    # Move the model to CPU.
    model.to(DEVICE)

    # Switch to evaluation mode.
    # This disables Dropout during inference and makes
    # the model produce consistent predictions.
    model.eval()

    return model


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================
# The uploaded image must undergo the same preprocessing
# used when evaluating the trained E4 model.
#
# 1. Resize the image to 224 x 224 pixels.
# 2. Convert the image into a PyTorch tensor.
# 3. Normalise RGB channels using ImageNet statistics.
#
# Data augmentation is not used during inference because
# the application should evaluate the original photograph
# using deterministic preprocessing.

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. MODEL PREDICTION FUNCTION
# ============================================================
# This function receives an uploaded bird photograph and
# returns the three most likely species predictions.
#
# The model produces raw classification scores (logits).
# Softmax converts these scores into probability values.
#
# torch.topk() selects the three classes with the highest
# predicted probabilities.
#
# torch.inference_mode() disables gradient calculations,
# reducing memory usage and speeding up inference.

def predict_bird(image, model, class_names):

    # Ensure that every image contains three RGB channels.
    image = image.convert("RGB")

    # Apply resizing, tensor conversion and normalisation.
    image_tensor = preprocess(image)

    # Add a batch dimension:
    # [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # Generate predictions without updating model weights.
    with torch.inference_mode():

        outputs = model(image_tensor)

        # Convert logits into class probabilities.
        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        # Find the three classes with the highest scores.
        scores, indices = torch.topk(
            probabilities,
            k=3
        )

    predictions = []

    # Convert numerical class indices into species names.
    for index, score in zip(
        indices.tolist(),
        scores.tolist()
    ):

        predictions.append({
            "Species": class_names[str(index)],
            "Confidence": score * 100
        })

    return predictions


# ============================================================
# 8. LOAD APPLICATION RESOURCES
# ============================================================
# Load the trained model and species-name mapping.
#
# If a required file is missing or the model architecture
# does not match the checkpoint, display the error and
# stop the application instead of producing wrong results.

try:
    class_names = load_class_names()
    model = load_model()

except Exception as error:
    st.error(
        "Unable to load the bird classification model."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 9. LEFT SIDEBAR
# ============================================================
# The sidebar provides information about the selected model.
#
# All performance numbers below come from the official
# experimental evaluation of the E4 model.
#
# These are fixed test-set results, not the confidence
# scores of an individual uploaded photograph.

with st.sidebar:

    st.markdown("""
    <div class="sidebar-label">
        COS30082 APPLIED MACHINE LEARNING
    </div>

    <div class="sidebar-title">
        Bird Species Classification
    </div>

    <div class="sidebar-subtitle">
        Model overview and performance
    </div>

    <div class="sidebar-label">
        MODEL DETAILS
    </div>

    <div class="sidebar-info">
        <span>Architecture</span>
        <strong>ResNet18</strong>
    </div>

    <div class="sidebar-info">
        <span>Pretraining</span>
        <strong>ImageNet-1K</strong>
    </div>

    <div class="sidebar-info">
        <span>Training strategy</span>
        <strong>Full Fine-Tuning</strong>
    </div>

    <div class="sidebar-info">
        <span>Bird species</span>
        <strong>200</strong>
    </div>

    <div class="sidebar-info">
        <span>Image size</span>
        <strong>224 × 224</strong>
    </div>
    """, unsafe_allow_html=True)

    # Add visual spacing between model details and metrics.
    st.markdown(
        "<div style='height:25px'></div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="sidebar-label">
        TEST PERFORMANCE
    </div>

    <div class="performance-card">
        <div class="performance-label">
            Top-1 Accuracy
        </div>
        <div class="performance-value">
            69.44%
        </div>
    </div>

    <div class="performance-card">
        <div class="performance-label">
            Average Accuracy per Class
        </div>
        <div class="performance-value">
            69.03%
        </div>
    </div>

    """, unsafe_allow_html=True)


# ============================================================
# 10. MAIN WEBSITE HEADER
# ============================================================
# The header introduces the application and explains
# its purpose to users who are unfamiliar with the project.

st.markdown("""
<div class="hero">
    <h1>Bird Species Classifier</h1>
    <p>
        Discover bird species using artificial intelligence.
        Upload a photograph and let our fine-tuned
        ResNet18 model analyse it.
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "This AI model recognises 200 bird species "
    "and displays the three most likely predictions."
)


# ============================================================
# 11. IMAGE UPLOAD AND PREDICTION INTERFACE
# ============================================================
# The main area is divided into two columns:
#
# Left column:
# - Image upload
# - Image preview
# - Identify Bird button
#
# Right column:
# - Predicted species
# - Model confidence
# - Top 3 predictions
# - Confidence comparison chart

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ------------------------------------------------------------
# 11.1 LEFT COLUMN - IMAGE UPLOAD
# ------------------------------------------------------------

with left_column:

    st.subheader("Upload Bird Image")

    # Accept common photograph formats.
    uploaded_file = st.file_uploader(
        "Choose a bird photograph",
        type=["jpg", "jpeg", "png"]
    )

    # The variable stays empty until an image is uploaded.
    image = None

    if uploaded_file is not None:

        try:

            # Open the uploaded image and convert it to RGB.
            image = Image.open(
                uploaded_file
            ).convert("RGB")

            # Display the original uploaded photograph.
            st.image(
                image,
                caption="Uploaded Bird Photograph",
                width="stretch"
            )

        except (UnidentifiedImageError, OSError):

            st.error(
                "This image cannot be opened. "
                "Please upload another JPG or PNG."
            )

    # Disable the button when no valid image is available.
    predict_button = st.button(
        "Identify Bird",
        type="primary",
        width="stretch",
        disabled=image is None
    )


# ------------------------------------------------------------
# 11.2 RIGHT COLUMN - PREDICTION RESULTS
# ------------------------------------------------------------

with right_column:

    st.subheader("Prediction Results")

    # Only perform inference when the user clicks the button
    # and a valid photograph is available.
    if predict_button and image is not None:

        with st.spinner(
            "Analysing bird photograph..."
        ):

            predictions = predict_bird(
                image,
                model,
                class_names
            )

        # The first prediction has the highest probability.
        best_prediction = predictions[0]

        best_species = best_prediction["Species"]
        best_confidence = best_prediction["Confidence"]

        # Display the most likely bird species in a card.
        st.markdown(
            f"""
            <div class="result-card">
                <p>Most Likely Bird Species</p>
                <h2>{best_species}</h2>
                <p>
                    Model confidence:
                    <b>{best_confidence:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display the top three results with progress bars.
        st.markdown("#### Top 3 Predictions")

        for rank, prediction in enumerate(
            predictions,
            start=1
        ):

            species = prediction["Species"]
            confidence = prediction["Confidence"]

            st.write(
                f"**{rank}. {species}** "
                f"— {confidence:.2f}%"
            )

            # st.progress() requires a value between 0 and 1.
            st.progress(
                min(max(confidence / 100, 0), 1)
            )

        # Create a horizontal bar chart comparing the
        # probabilities of the three predicted species.
        st.markdown("#### Confidence Comparison")

        chart_df = pd.DataFrame(predictions)

        st.bar_chart(
            chart_df.set_index("Species"),
            y="Confidence",
            horizontal=True
        )

        st.caption(
            "Confidence values are softmax scores "
            "and are not guarantees of correctness."
        )

    else:

        # Display instructions before a prediction is made.
        st.info(
            "Upload a bird photograph and click "
            "'Identify Bird' to view predictions."
        )




# ============================================================
# 12. IMPORTANT LIMITATIONS
# ============================================================
# Display the model limitations inside a styled card.
# This section only changes the website appearance.

st.divider()

st.markdown("""
<div class="limitation-box">
<div class="limitation-title">Important Limitations</div>
<p class="limitation-text">
This application is developed for educational purposes
and can only classify the 200 bird species included in
its training dataset. Images containing unfamiliar species,
poor lighting, blur, or no birds may result in incorrect
predictions. Confidence scores represent the model's
estimated probabilities and do not guarantee prediction
accuracy.
</p>
<div class="limitation-footer">
Bird Species Classification | Applied Machine Learning Assignment
</div>
</div>
""", unsafe_allow_html=True)

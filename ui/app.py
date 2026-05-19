import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO


st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered"
)

st.title("🩺 AI Pneumonia Detection System")

st.markdown(
    """
Upload a chest X-ray image to detect:
- NORMAL lungs
- PNEUMONIA

Includes:
✅ Deep Learning Prediction  
✅ Confidence Score  
✅ Explainable AI Heatmap (GradCAM)
"""
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    files = {
        "file": uploaded_file.getvalue()
    }

    try:

        with st.spinner("Analyzing X-ray..."):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

        result = response.json()

        # Handle backend errors
        if "error" in result:

            st.error(result["error"])

        else:

            prediction = result["prediction"]

            confidence = result["confidence"]

            heatmap_base64 = result["heatmap"]

            st.success(
                f"Prediction: {prediction}"
            )

            st.info(
                f"Confidence: {confidence:.4f}"
            )

            # Decode heatmap
            heatmap_bytes = base64.b64decode(
                heatmap_base64
            )

            heatmap_image = Image.open(
                BytesIO(heatmap_bytes)
            )

            st.image(
                heatmap_image,
                caption="GradCAM Heatmap",
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Error: {e}")
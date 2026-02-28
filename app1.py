import streamlit as st
from paddleocr import PaddleOCR
import os
import glob

# ----- UI Design Only (No Logic Changes) -----
st.set_page_config(page_title="PaddleOCR Colab", page_icon="📄", layout="centered")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at top, #2b1d24 0%, #120e15 100%);
        font-family: 'Inter', sans-serif;
        color: #e6edf3;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(135deg, #ff7b00, #ff007b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Upload Box */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed rgba(255, 123, 0, 0.5);
        border-radius: 16px;
        background-color: rgba(43, 29, 36, 0.6);
        transition: all 0.3s ease-in-out;
        padding: 2rem;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #ff007b;
        background-color: rgba(43, 29, 36, 0.9);
        transform: translateY(-2px);
    }
    
    /* Main Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #ff7b00 0%, #ff007b 100%);
        color: #ffffff;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 123, 0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 0, 123, 0.5);
    }
    
    /* Image Containers */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------

@st.cache_resource
def load_ocr():
    return PaddleOCR(
        use_doc_orientation_classify=False,  # stable in Colab
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

ocr = load_ocr()

st.title("📄 PaddleOCR Streamlit (Colab + ngrok)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Save uploaded image
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Run OCR"):
        with st.spinner("Running OCR..."):
            result = ocr.predict(image_path)

        # Let PaddleOCR save outputs
        for res in result:
            res.save_to_img("output")

        # Find OCR result image (png or jpg)
        img_files = sorted(
            glob.glob("output/*_ocr_res_img.png") +
            glob.glob("output/*_ocr_res_img.jpg")
        )

        if img_files:
            st.success("OCR completed!")

            st.subheader("OCR Result Image")
            st.image(img_files[-1], use_container_width=True)
        else:
            st.error("OCR image not found in output folder")
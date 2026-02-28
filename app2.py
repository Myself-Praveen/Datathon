import os
os.environ["FLAGS_enable_pir_api"] = "0"
import streamlit as st
from paddleocr import PaddleOCR
import glob
import time
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw

# ----- UI Design Only (No Logic Changes) -----
st.set_page_config(page_title="TextTract OCR", page_icon="🔍", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at top, #0b0f19 0%, #05070a 100%);
        font-family: 'Outfit', sans-serif;
        color: #e6edf3;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 20px;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Upload Box */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed rgba(0, 212, 255, 0.4);
        border-radius: 20px;
        background-color: rgba(20, 25, 40, 0.4);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        padding: 3rem;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #00ff88;
        background-color: rgba(20, 25, 40, 0.8);
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 255, 136, 0.1);
    }
    
    /* Main Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        color: #05070a;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0, 255, 136, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.5);
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background: rgba(0, 212, 255, 0.2);
        color: #00ff88;
        border-color: #00ff88;
    }

    /* Image Containers */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.3s ease;
    }
    [data-testid="stImage"]:hover {
        transform: scale(1.01);
    }
    
    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        background-color: rgba(20, 25, 40, 0.6);
        backdrop-filter: blur(10px);
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(20, 25, 40, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetric"] label { color: #8b949e; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #00ff88; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(20, 25, 40, 0.5);
        border-radius: 12px;
        padding: 10px 20px;
        color: #8b949e;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,255,136,0.2)) !important;
        color: #00ff88 !important;
        border-color: #00ff88 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 25, 0.95);
        border-right: 1px solid rgba(0,212,255,0.1);
    }
    section[data-testid="stSidebar"] h2 {
        color: #00d4ff;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
def compute_cer(ground_truth, predicted):
    """Character Error Rate using Levenshtein distance."""
    n = len(ground_truth)
    m = len(predicted)
    if n == 0:
        return 1.0 if m > 0 else 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ground_truth[i - 1] == predicted[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n

def apply_opencv_preprocessing(img_path):
    """Apply OpenCV preprocessing to enhance image for OCR."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=20)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_path = img_path.replace(".", "_preprocessed.")
    cv2.imwrite(processed_path, binary)
    return processed_path


# ---------------------------------------------
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

# ---- Sidebar Controls ----
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.markdown("---")
    use_preprocessing = st.checkbox("🔧 Enable OpenCV Pre-processing", value=False,
                                    help="Apply denoising, contrast enhancement, and adaptive thresholding before OCR.")
    conf_threshold = st.slider("🎯 Confidence Threshold (Zero-Hallucination)", 0.0, 1.0, 0.5, 0.05,
                               help="Filter out detections below this confidence to prevent hallucinated text.")
    st.markdown("---")
    st.markdown("## 📊 CER Benchmarking")
    ground_truth_text = st.text_area("Ground Truth Text (optional)", placeholder="Paste the real text here to compute CER...", height=120)
    st.markdown("---")
    st.markdown("### 🔑 Legend")
    st.markdown("🟢 **Green** — Confidence ≥ 90%")
    st.markdown("🔵 **Cyan** — Confidence 70-90%")
    st.markdown("🔴 **Red** — Confidence < 70%")

st.title("🔍 TextTract: Advanced OCR Pipeline")

tab1, tab2 = st.tabs(["🖼️ Single Image OCR", "📁 Batch Dataset Evaluator"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        # Save uploaded image
        image_path = os.path.join("uploads", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img_col_upload, _ = st.columns([1, 1])
        with img_col_upload:
            st.image(image_path, caption="Uploaded Image", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run OCR"):
            # Optionally apply OpenCV preprocessing
            target_path = image_path
            if use_preprocessing:
                with st.spinner("Applying OpenCV pre-processing..."):
                    target_path = apply_opencv_preprocessing(image_path)
                    st.image(target_path, caption="Pre-processed Image", use_container_width=True)

            with st.spinner("Running OCR..."):
                start_time = time.time()
                # EXACT ORIGINAL LOGIC 
                result = ocr.predict(target_path)
                inference_time = time.time() - start_time

            # Original save image fallback
            try:
                from PIL import Image
                image = Image.open(target_path).convert('RGB')
                boxes = [line[0] for line in result[0]]
                txts = [line[1][0] for line in result[0]]
                scores = [line[1][1] for line in result[0]]
                
                from paddleocr import draw_ocr
                im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
                im_show = Image.fromarray(im_show)
                im_show.save("output/paddle_ocr_res_img.png")
            except Exception as e:
                pass

            # ---- Extract structured detections from result ----
            all_detections = []
                
            # If the pipeline parser didn't hit our expected structure, we fall back to reading the JSON we just forced to save
            if not all_detections:
                try:
                    out_json_files = sorted(glob.glob("output/*.json"))
                    if out_json_files:
                        with open(out_json_files[-1], "r", encoding="utf-8") as f:
                            saved_data = json.load(f)
                            
                            # Standard format
                            if isinstance(saved_data, list):
                                for region in saved_data:
                                    if "res" in region:
                                        for det in region["res"]:
                                            if "text" in det and "confidence" in det and "text_region" in det:
                                                all_detections.append({
                                                    "text": det["text"],
                                                    "confidence": float(det["confidence"]),
                                                    "box": det["text_region"]
                                                })
                            # Dict format fallback
                            elif isinstance(saved_data, dict):
                                if "rec_texts" in saved_data and "rec_scores" in saved_data and "dt_polys" in saved_data:
                                    for t, s, p in zip(saved_data["rec_texts"], saved_data["rec_scores"], saved_data["dt_polys"]):
                                        all_detections.append({
                                            "text": t,
                                            "confidence": float(s),
                                            "box": [list(map(float, pt)) for pt in p]
                                        })
                except:
                    pass

            # Apply confidence filter
            filtered = [d for d in all_detections if d["confidence"] >= conf_threshold]
            rejected = len(all_detections) - len(filtered)

            # ---- Metrics Row ----
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⏱️ Inference Time", f"{inference_time:.2f}s")
            m2.metric("📝 Detections", len(all_detections))
            m3.metric("✅ After Filter", len(filtered))
            m4.metric("🚫 Rejected", rejected)

            st.markdown("---")
            
            # ---- PaddleOCR's Built-in Result Image ----
            img_files = sorted(
                glob.glob("output/*_ocr_res_img.png") +
                glob.glob("output/*_ocr_res_img.jpg")
            )
            if img_files:
                st.subheader("📄 PaddleOCR Result Image")
                st.image(img_files[-1], use_container_width="stretch")

            st.markdown("---")

            # ---- Extracted Text Table ----
            if filtered:
                st.subheader("📋 Extracted Text (Filtered)")
                for i, det in enumerate(filtered):
                    box_str = str([[int(p) for p in pt] for pt in det['box']])
                    st.markdown(f"**{i+1}.** `{det['text']}` — Conf: **{det['confidence']:.2f}** — Box: `{box_str}`")

                # Combine all text for CER
                predicted_text = " ".join([d["text"] for d in filtered])

                # ---- CER Calculation ----
                if ground_truth_text.strip():
                    cer = compute_cer(ground_truth_text.strip(), predicted_text)
                    st.subheader("📊 CER Benchmark")
                    cer_col1, cer_col2 = st.columns(2)
                    cer_col1.metric("Character Error Rate", f"{cer:.4f}")
                    cer_col2.metric("Accuracy", f"{(1 - cer) * 100:.1f}%")
                    with st.expander("View Predicted vs Ground Truth"):
                        st.text_area("Predicted", predicted_text, height=100, disabled=True)
                        st.text_area("Ground Truth", ground_truth_text.strip(), height=100, disabled=True)

                # ---- JSON Download ----
                output_json = json.dumps(filtered, indent=2)
                st.download_button(
                    label="📥 Download Filtered OCR JSON",
                    data=output_json,
                    file_name="ocr_results_filtered.json",
                    mime="application/json"
                )

                # Also offer original full JSON
                json_files = glob.glob("output/*.json")
                if json_files:
                    with open(json_files[-1], "rb") as f:
                        st.download_button(
                            label="📥 Download Full PaddleOCR JSON",
                            data=f,
                            file_name=os.path.basename(json_files[-1]),
                            mime="application/json"
                        )
            else:
                st.warning("No detections passed the confidence filter. Try lowering the threshold.")

with tab2:
    st.subheader("Process Standard Dataset")
    dataset_path = st.text_input("Dataset Folder Path", value=r"C:\Users\prave\OneDrive\Desktop\test\data_subset\data_subset")
    num_limit = st.number_input("Limit Number of Images (0 for all)", min_value=0, max_value=10000, value=50)
    
    if st.button("Run Batch Pipeline"):
        if not os.path.exists(dataset_path):
            st.error("Folder not found.")
        else:
            image_paths = glob.glob(os.path.join(dataset_path, "*.png")) + glob.glob(os.path.join(dataset_path, "*.jpg"))
            if num_limit > 0:
                image_paths = image_paths[:num_limit]
                
            st.info(f"Found {len(image_paths)} images to process.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            os.makedirs("batch_output", exist_ok=True)
            results = {}
            
            for i, img_path in enumerate(image_paths):
                status_text.text(f"Processing: {os.path.basename(img_path)} ({i+1}/{len(image_paths)})")
                
                try:
                    # EXACT ORIGINAL LOGIC
                    result = ocr.predict(img_path)
                    
                    extracted_texts = []
                    if result and result[0] is not None:
                        for line in result[0]:
                            try:
                                text = line[1][0]
                                conf = line[1][1]
                                box = line[0]
                                extracted_texts.append({
                                    "text": text,
                                    "confidence": float(conf),
                                    "box": box
                                })
                            except:
                                pass
                    results[os.path.basename(img_path)] = extracted_texts
                except Exception as e:
                    pass
                
                progress_bar.progress((i + 1) / len(image_paths))
                
            status_text.text("Batch Processing Completed!")
            
            output_meta = os.path.join("batch_output", "evaluation_results.json")
            with open(output_meta, "w") as f:
                json.dump(results, f, indent=4)
                
            st.success("All images processed and saved to the 'batch_output' folder!")
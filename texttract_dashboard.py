import streamlit as st
from paddleocr import PaddleOCR
import os
# Fix PaddlePaddle bug on Windows CPU
os.environ["FLAGS_enable_pir_api"] = "0"
import time
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextTract – End-to-End Local OCR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium Dark CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background and typography */
    .stApp {
        background: radial-gradient(circle at top left, #1b203d 0%, #0d1117 100%);
        color: #e6edf3;
    }
    
    /* Hero Banner */
    .hero {
        text-align: center;
        padding: 3rem 1rem 2rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(0, 255, 136, 0.15);
        position: relative;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .hero p {
        color: #8b949e;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 10px;
    }
    
    /* Tags */
    .tags {
        display: flex; gap: 10px; justify-content: center; margin-top: 15px;
    }
    .tag {
        padding: 5px 15px; border-radius: 50px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }
    .tag-local { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid rgba(0, 255, 136, 0.3); }
    .tag-zero { background: rgba(0, 212, 255, 0.1); color: #00d4ff; border: 1px solid rgba(0, 212, 255, 0.3); }

    /* Metric Cards */
    .metric-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(30, 35, 50, 0.5); border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px; padding: 20px; transition: all 0.3s ease;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .metric-container:hover {
        border-color: rgba(0, 255, 136, 0.3); transform: translateY(-5px);
    }
    .metric-val { font-size: 2.2rem; font-weight: 800; color: #00ff88; margin-bottom: 5px; }
    .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; }

    /* Subheaders */
    h3 { color: #ffffff !important; font-weight: 600 !important; margin-top: 2rem !important; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }

    /* Upload Box */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed rgba(0, 212, 255, 0.5); border-radius: 16px;
        background-color: rgba(30, 35, 50, 0.6); transition: all 0.3s ease-in-out; padding: 2rem;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #00ff88; background-color: rgba(30, 35, 50, 0.9);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        color: #0d1117; font-weight: 700; border: none; border-radius: 12px;
        padding: 0.8rem; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        width: 100%; text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 255, 136, 0.4);
    }

    /* Image Display */
    [data-testid="stImage"] {
        border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize Core OCR Engine ────────────────────────────────────────────────
@st.cache_resource
def load_ocr():
    # Strict Local Pipeline (Zero-Cloud constraint)
    return PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )

ocr = load_ocr()

# ── Helper Functions ──────────────────────────────────────────────────────────

# Optional Task: OpenCV Pre-processing
def preprocess_image_opencv(image_path):
    """Applies advanced OpenCV pre-processing for noisy/degraded images"""
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise (Non-local Means Denoising)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Enhance contrast
    alpha = 1.3 # Contrast control
    beta = 10   # Brightness control
    adjusted = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
    
    # Save processed image
    processed_path = image_path.replace(".", "_processed.")
    cv2.imwrite(processed_path, adjusted)
    return processed_path

# Drawing Spatial Bounding Boxes
def draw_spatial_boxes(image: Image.Image, detections, box_color="#00FF88", font_size=16):
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fallback to default font if arial fails
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except: font = ImageFont.load_default()

    for det in detections:
        poly = [(int(p[0]), int(p[1])) for p in det["box"]]
        # Draw Box Polygon
        for i in range(len(poly)):
            draw.line([poly[i], poly[(i + 1) % len(poly)]], fill=box_color, width=3)
        
        # Draw Text Label & Confidence background
        label = f"{det['text']} ({det['confidence']:.2f})"
        lx, ly = min(p[0] for p in poly), min(p[1] for p in poly) - font_size - 4
        if ly < 0: ly = max(p[1] for p in poly) + 2
        
        bbox = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill=(0, 0, 0, 180))
        draw.text((lx, ly), label, fill="#FFFFFF", font=font)

    return Image.alpha_composite(annotated, overlay).convert("RGB")

# Character Error Rate (CER) Metric
def compute_cer(gt, pred):
    """Computes exact Character Error Rate between Ground Truth and Prediction"""
    gt, pred = " ".join(gt.lower().split()), " ".join(pred.lower().split())
    if len(gt) == 0: return 0.0 if len(pred) == 0 else float(len(pred))
    
    # Levenshtein Distance
    prev = list(range(len(pred) + 1))
    for i, c1 in enumerate(gt):
        curr = [i + 1]
        for j, c2 in enumerate(pred):
            curr.append(min(curr[j]+1, prev[j+1]+1, prev[j]+(c1 != c2)))
        prev = curr
    return round(prev[-1] / len(gt), 4)


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>TextTract Challenge</h1>
    <p>End-to-end Local OCR Pipeline &nbsp;·&nbsp; Spatial Detection &nbsp;·&nbsp; Zero Hallucination</p>
    <div class="tags">
        <span class="tag tag-local">🛡️ 100% Local Inference</span>
        <span class="tag tag-zero">☁️ Zero-Cloud Constraint</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Configurations ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PaddlePaddle_Logo.svg/512px-PaddlePaddle_Logo.svg.png", width=150)
    st.markdown("### ⚙️ Pipeline Settings")
    st.markdown("---")
    
    st.markdown("#### Anti-Hallucination Filter")
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.70, 0.05, 
                            help="Zero Hallucination Constraint: Drops uncertain generative guesses to strictly read what's visibly present.")
    
    st.markdown("---")
    st.markdown("#### Computer Vision (Optional)")
    use_opencv = st.checkbox("Enable OpenCV Pre-processing", help="Applies Grayscale, Non-Local Means Denoising, and Contrast adjustments before OCR.")
    
    st.markdown("---")
    st.markdown("#### Verification Matrix")
    eval_mode = st.checkbox("Enable Ground Truth Evaluation", help="Allows calculation of Character Error Rate (CER) against your actual text.")

# ── Main Application Board ────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📥 Upload Test Image (Street sign, Document, License Plate, etc.)", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Pre-processing Check
    if use_opencv:
        st.info("🪄 Applying OpenCV Image Pre-processing (Grayscale + Denoise)...")
        target_path = preprocess_image_opencv(temp_path)
    else:
        target_path = temp_path
        
    original_pil = Image.open(target_path).convert("RGB")

    # Run Prediction
    if st.button("🚀 Execute OCR Pipeline"):
        with st.spinner("Executing Local text detection & recognition..."):
            start_time = time.perf_counter()
            # Raw PaddleOCR run
            raw_result = list(ocr.predict(target_path))
            inference_time = round(time.perf_counter() - start_time, 3)
            
            # Anti-hallucination Post-processing
            valid_detections = []
            if raw_result:
                res = raw_result[0]
                # Fallback dictionary structures if properties vary
                polys = res.get("dt_polys", res.get("rec_polys", []))
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                
                # Zip and filter
                for i in range(min(len(polys), len(texts), len(scores))):
                    conf = scores[i]
                    if conf >= conf_thresh:
                        valid_detections.append({
                            "box": polys[i], "text": texts[i], "confidence": conf
                        })

            st.session_state["pipeline_results"] = {
                "detections": valid_detections,
                "time": inference_time,
                "img_path": target_path
            }

    # ── Render Results ──────────────
    if "pipeline_results" in st.session_state:
        res = st.session_state["pipeline_results"]
        detections = res["detections"]
        
        # Key Metrics Banner
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-container"><div class="metric-val">{res["time"]}s</div><div class="metric-label">Total Inference Time</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><div class="metric-val">{len(detections)}</div><div class="metric-label">Valid Box Detections</div></div>', unsafe_allow_html=True)
        with col3:
            avg_conf = sum(d['confidence'] for d in detections)/len(detections) if detections else 0
            st.markdown(f'<div class="metric-container"><div class="metric-val">{avg_conf:.1%}</div><div class="metric-label">Pipeline Certainty</div></div>', unsafe_allow_html=True)

        # Visualizations (Side by Side)
        st.markdown("### 👁️ Spatial Visualization")
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("**Original Target Image**")
            st.image(original_pil, use_container_width=True)
            
        with img_col2:
            st.markdown("**Local Bounding Boxes overlay**")
            if detections:
                annotated_img = draw_spatial_boxes(original_pil, detections)
                st.image(annotated_img, use_container_width=True)
            else:
                st.warning("No text detected above the confidence threshold.")

        # Data & JSON Output
        st.markdown("### 📊 Extracted Text Mappings")
        tab1, tab2 = st.tabs(["📝 Detailed Box List", "🔧 Developer JSON"])
        
        with tab1:
            if detections:
                markdown_table = "| ID | Recognized Text | Confidence % | Bounding Box [Top Left] | [Bottom Right] |\n|---|---|---|---|---|\n"
                for idx, d in enumerate(detections, 1):
                    tl, br = d["box"][0], d["box"][2]
                    tl_str, br_str = f"({int(tl[0])}, {int(tl[1])})", f"({int(br[0])}, {int(br[1])})"
                    markdown_table += f"| {idx} | **{d['text']}** | {d['confidence']:.2%} | `{tl_str}` | `{br_str}` |\n"
                st.markdown(markdown_table)
            else:
                st.info("Empty extraction. Adjust Anti-Hallucination threshold if needed.")
                
        with tab2:
            st.code(json.dumps(detections, indent=2), language="json")

        # CER Benchmark
        if eval_mode:
            st.markdown("### 📐 Character Error Rate (CER) Benchmarking")
            st.info("Input the absolute ground truth of the image here to calculate performance accuracy.")
            gt_text = st.text_area("Ground Truth Text:")
            if gt_text:
                predicted_text = " ".join([d["text"] for d in detections])
                cer_score = compute_cer(gt_text, predicted_text)
                accuracy_color = "#00ff88" if cer_score < 0.10 else "#ffcc00" if cer_score < 0.25 else "#ff3366"
                
                st.markdown(f"""
                <div style="background: rgba(30, 35, 50, 0.5); padding: 20px; border-radius: 12px; margin-top: 10px; border-left: 5px solid {accuracy_color}">
                    <h4 style="margin:0; color: #8b949e">Computed CER Score</h4>
                    <span style="font-size: 2rem; font-weight: 800; color: {accuracy_color}">{cer_score:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Actual Extracted String:** \n> *{predicted_text}*")

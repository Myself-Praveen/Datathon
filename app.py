"""
app.py  –  TextTract Challenge: Interactive Streamlit Web Dashboard
================================================================
Premium dark-themed OCR dashboard  ·  PaddleOCR 2.9  ·  Zero Cloud APIs
"""

import os
import time
import json
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextTract – Local OCR Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(145deg, #0a0a0f 0%, #111127 50%, #0d1117 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(0, 255, 136, 0.1);
    }

    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        margin-bottom: 1rem;
    }
    .hero-header h1 {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem; letter-spacing: -1px;
    }
    .hero-header p { color: #8892b0; font-size: 1.05rem; font-weight: 300; }

    .metric-card {
        background: linear-gradient(145deg, rgba(20,20,40,0.8), rgba(30,30,60,0.6));
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 16px; padding: 1.2rem 1.4rem;
        text-align: center; backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(0,255,136,0.12);
    }
    .metric-card .metric-value {
        font-size: 2rem; font-weight: 700; color: #00ff88; margin-bottom: 0.1rem;
    }
    .metric-card .metric-label {
        font-size: 0.85rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px;
    }

    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #e6f1ff;
        margin: 1.5rem 0 0.8rem; padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,255,136,0.2);
    }

    .status-badge {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 50px;
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .status-badge.local {
        background: rgba(0,255,136,0.15); color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    .status-badge.zero-cloud {
        background: rgba(124,58,237,0.15); color: #a78bfa;
        border: 1px solid rgba(124,58,237,0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00d4ff) !important;
        color: #0a0a0f !important; font-weight: 700 !important;
        border: none !important; border-radius: 12px !important;
        padding: 0.7rem 2rem !important; font-size: 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important; letter-spacing: 1px !important;
    }
    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 24px rgba(0,255,136,0.3) !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(20,20,40,0.5);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
    }

    .footer-text {
        text-align: center; color: #4a5568; font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# OCR ENGINE  (PaddleOCR 2.9.x API — uses ocr.ocr() not ocr.predict())
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_ocr():
    """Load PaddleOCR v4 models (lightweight, Windows-stable)."""
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
    )


def parse_ocr_result(raw_result, confidence_threshold=0.50):
    """Parse PaddleOCR 2.x output → list of dicts with zero-hallucination filter."""
    detections = []
    if not raw_result or not raw_result[0]:
        return detections

    for line in raw_result[0]:
        box = line[0]                # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]            # recognised string
        conf = float(line[1][1])     # confidence score

        if conf < confidence_threshold:
            continue  # ← Zero-hallucination: drop low-confidence guesses

        detections.append({
            "box": box,
            "text": text,
            "confidence": round(conf, 4),
        })
    return detections


def draw_boxes(image: Image.Image, detections, box_color="#00FF88",
               font_size=14, show_confidence=True):
    """Draw bounding-box polygons and text labels on the image."""
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    for det in detections:
        poly = [(int(p[0]), int(p[1])) for p in det["box"]]
        # outline
        for i in range(len(poly)):
            draw.line([poly[i], poly[(i + 1) % len(poly)]],
                      fill=box_color, width=2)
        # label
        label = det["text"]
        if show_confidence:
            label += f" ({det['confidence']:.0%})"
        lx = min(p[0] for p in poly)
        ly = min(p[1] for p in poly) - font_size - 4
        if ly < 0:
            ly = max(p[1] for p in poly) + 2
        bbox = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1],
                        fill=(0, 0, 0, 180))
        draw.text((lx, ly), label, fill="#FFFFFF", font=font)

    return Image.alpha_composite(annotated, overlay).convert("RGB")


# ── CER computation ──────────────────────────────────────────────────────────

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j]+1, prev[j+1]+1, prev[j]+(c1 != c2)))
        prev = curr
    return prev[-1]


def compute_cer(gt, pred):
    gt = " ".join(gt.lower().split())
    pred = " ".join(pred.lower().split())
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else float(len(pred))
    return round(levenshtein(gt, pred) / len(gt), 4)


# ═════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <h1>🔍 TextTract</h1>
    <p>End-to-end local OCR pipeline &nbsp;·&nbsp; PaddleOCR &nbsp;·&nbsp; Zero Hallucination</p>
    <div style="margin-top: 0.8rem;">
        <span class="status-badge local">🟢 100% Local Inference</span>&nbsp;&nbsp;
        <span class="status-badge zero-cloud">☁️ Zero Cloud APIs</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    st.markdown("---")

    st.markdown("**Zero-Hallucination Control**")
    confidence_threshold = st.slider(
        "Min Confidence Threshold", 0.0, 1.0, 0.50, 0.05,
        help="Detections below this confidence are **dropped**.",
    )

    st.markdown("---")
    st.markdown("**Visualization**")
    box_color = st.color_picker("Bounding Box Color", "#00FF88")
    show_confidence_labels = st.checkbox("Show confidence on boxes", True)
    font_size = st.slider("Label font size", 8, 28, 14)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN  –  Upload & OCR
# ═════════════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📤  Upload an image for OCR",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
)

if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    original_image = Image.open(image_path).convert("RGB")

    st.markdown('<div class="section-header">📷 Uploaded Image</div>',
                unsafe_allow_html=True)
    st.image(original_image, use_container_width=True)

    if st.button("🚀  Run OCR Pipeline", use_container_width=True):
        ocr = load_ocr()

        with st.spinner("⏳ Running PaddleOCR locally..."):
            t0 = time.perf_counter()
            raw = ocr.ocr(image_path, cls=True)      # PaddleOCR 2.x API
            inference_time = round(time.perf_counter() - t0, 3)

        detections = parse_ocr_result(raw, confidence_threshold)

        st.session_state["ocr_result"] = {
            "detections": detections,
            "inference_time_s": inference_time,
        }
        st.session_state["image_path"] = image_path

    # ── Display results ──────────────────────────────────────────────────
    if "ocr_result" in st.session_state:
        result = st.session_state["ocr_result"]
        detections = result["detections"]
        confs = [d["confidence"] for d in detections] if detections else [0]

        # Metrics
        st.markdown('<div class="section-header">📊 Pipeline Metrics</div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{result['inference_time_s']}s</div>
                <div class="metric-label">Inference Time</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(detections)}</div>
                <div class="metric-label">Detections</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            avg_c = sum(confs) / len(confs) if confs else 0
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_c:.0%}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)

        # Side-by-side images
        st.markdown('<div class="section-header">🖼️ Bounding Boxes</div>',
                    unsafe_allow_html=True)
        annotated = draw_boxes(
            original_image, detections,
            box_color=box_color, font_size=font_size,
            show_confidence=show_confidence_labels,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.image(original_image, use_container_width=True)
        with c2:
            st.markdown("**Annotated**")
            st.image(annotated, use_container_width=True)

        # Text output
        st.markdown('<div class="section-header">📝 Extracted Text</div>',
                    unsafe_allow_html=True)
        tab_table, tab_json = st.tabs(["📋 Table", "🔧 JSON"])
        with tab_table:
            if detections:
                lines = ["| # | Text | Confidence | Top-Left |",
                         "|---|------|------------|----------|"]
                for i, d in enumerate(detections, 1):
                    tl = d["box"][0]
                    lines.append(
                        f"| {i} | {d['text']} | {d['confidence']:.2%} "
                        f"| ({int(tl[0])}, {int(tl[1])}) |"
                    )
                st.markdown("\n".join(lines))
            else:
                st.info("No text detected above the confidence threshold.")
        with tab_json:
            st.code(json.dumps(detections, indent=2, ensure_ascii=False),
                    language="json")

        # CER panel
        st.markdown('<div class="section-header">📐 CER Evaluation</div>',
                    unsafe_allow_html=True)
        with st.expander("🧪 Benchmark against Ground Truth"):
            gt_input = st.text_area("Ground Truth Text", height=120,
                                    placeholder="Paste ground truth here…")
            if gt_input:
                pred_text = " ".join(d["text"] for d in detections)
                cer = compute_cer(gt_input, pred_text)
                quality = ("🟢 Excellent" if cer < 0.05
                           else "🟡 Good" if cer < 0.15
                           else "🟠 Fair" if cer < 0.30
                           else "🔴 Poor")
                cc1, cc2 = st.columns(2)
                with cc1:
                    color = '#00ff88' if cer < 0.15 else '#ff6b6b'
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value" style="color:{color}">{cer:.2%}</div>
                        <div class="metric-label">Character Error Rate</div>
                    </div>""", unsafe_allow_html=True)
                with cc2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem">{quality}</div>
                        <div class="metric-label">Quality</div>
                    </div>""", unsafe_allow_html=True)
                st.code(pred_text, language="text")
else:
    st.markdown("""<div style="
        text-align:center; padding:4rem 2rem;
        background:linear-gradient(145deg,rgba(20,20,40,0.4),rgba(30,30,60,0.3));
        border:2px dashed rgba(0,255,136,0.2); border-radius:20px; margin:2rem 0;
    ">
        <p style="font-size:3rem; margin-bottom:0.5rem;">📄</p>
        <p style="color:#8892b0; font-size:1.1rem;">
            Drag & drop an image above to begin OCR extraction
        </p>
        <p style="color:#4a5568; font-size:0.85rem; margin-top:0.5rem;">
            Supported: JPG · PNG · BMP · TIFF · WEBP
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("""<div class="footer-text">
    TextTract &nbsp;·&nbsp; Torch Titans &nbsp;·&nbsp; 100% Local &nbsp;·&nbsp; PaddleOCR &nbsp;·&nbsp; Zero Cloud APIs
</div>""", unsafe_allow_html=True)

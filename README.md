<h1 align="center">🔍 TextTract: Advanced Local OCR Pipeline</h1>

<div align="center">
  <p><strong>End-to-end Local OCR Pipeline · Spatial Detection · Zero Hallucination</strong></p>
  <p>Built for the <b>TextTract Challenge</b></p>
</div>

---

## 📖 Overview

Extracting text from noisy environments—such as scanned documents, street signs, or license plates—involves complex spatial detection without the generative guessing ("hallucinations") commonly found in massive cloud LLMs. 

**TextTract** is an end-to-end Optical Character Recognition (OCR) pipeline built to solve this exact problem. Adhering strictly to the challenge's **Zero-Cloud constraint**, this pipeline processes images entirely locally without hitting any arbitrary cloud LLM endpoints (like GPT-4 Vision). 

It reads exactly what is visible, provides pin-point bounding box coordinates, evaluates Character Error Rate (CER), and is deployed as an immersive, interactive Streamlit dashboard.

---

## ✨ Hackathon Core Objectives Met

### 1. 🛡️ The Local OCR Pipeline (Zero-Cloud Constraint)
* **Local Processing:** Utilizes [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to perform heavy lifting locally on CPU/GPU hardware without any external API calls.
* **Spatial & Text Recognition:** Returns precise polygon bounding boxes alongside extracted text arrays.
* **Optional Task Completed:** Features a custom OpenCV pre-processing toggle that applies Grayscale conversion, Non-Local Means Denoising, and Contrast adjustments to rescue degraded images before OCR execution.

### 2. 🎯 Spatial Accuracy & Zero Hallucination
* **Zero Hallucination Guarantee:** A built-in user-configurable **Confidence Threshold filter**. If an image is degraded, unsure generative guesses are aggressively dropped to prevent hallucinatory auto-correction and strictly output the exact visible text.
* **CER Benchmarking:** Features a dedicated benchmarking module. Users can supply ground truth text to automatically compute exact **Character Error Rates (CER)** and accuracy percentage scores using mathematical Levenshtein distance calculations.

### 3. 🖥️ Interactive Web Dashboard
* **Framework:** High-performance, premium UI dashboard built entirely on **Streamlit**.
* **Visualizations:** Generates spatial visualizations showing strict OCR bounding polygons overlaid on the uploaded target images.
* **Data Extraction:** Presents a detailed formatted table mapping all valid text strings to their exact pixel coordinates and pipeline confidence vectors.
* **Output Exporting:** Supports instant downloading of the resulting JSON developer schema for broader application integration.
* **Metrics:** Tracks core statistics like specific Inference Processing Time, Valid Detections, and count of Hallucinations Rejected by your filter.
* **Batch Evaluator:** Includes a massive secondary tab strictly dedicated to pointing towards entire massive unlabelled dataset folders, sequentially compiling them locally without touching the web.

---

## 🚀 Tech Stack

### Frontend & Dashboard
* **Streamlit:** Serves as the interactive UI and high-level routing framework.
* **Custom CSS / Glassmorphism:** Custom aesthetic enhancements, premium dark modes, gradients, dynamic UI styling.

### Computer Vision & ML
* **PaddlePaddle & PaddleOCR (`ch_PP-OCRv4`):** Providing the bleeding edge framework for text line orientation and deep multi-language detection.
* **OpenCV (`cv2`):** Used strictly for raw spatial manipulation (Gaussian Blurs, Adaptive Thresholding, Denoising).
* **Pillow (`PIL`):** Spatial image buffering and drawing bounding boxes directly into bitmap image renderings.

### Mathematical Implementations
* **Numpy:** For complex array structures backing the bounding polygon coordinates.
* **Levenshtein Algorithm:** Implemented natively in Python to iterate through prediction versus ground-truth strings for exact character-level error percentages.

---

## 📂 Project Structure

```text
Datathon/
├── app2.py                 # The centralized main UI App Dashboard
├── texttract_dashboard.py  # Alternative pipeline architecture design UI
├── batch_eval.py           # Headless local script specifically for benchmarking datasets
├── requirements.txt        # Crucial Python environment dependencies 
├── .gitignore              # Restricting environment file pollution on Github
├── datathon.ipynb          # Jupyter Notebooks containing the POC research environments
├── iam.ipynb               
├── sample.jpg              # Default test media file       
├── output/                 # Dynamically generated inference JSONs & Bounding Box pictures
├── batch_output/           # Output directory reserved exclusively for processing large unstructured folders
└── uploads/                # Short-term UI buffering directory for incoming image evaluations
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Myself-Praveen/Datathon.git
cd Datathon
```

**2. Create a virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate   
# On Unix/MacOS:
# source venv/bin/activate 
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Note for evaluators: By default, PaddlePaddle may demand CUDA hardware libraries. To run this pipeline purely on regular CPU hardware, install the strict CPU version of `paddlepaddle` first).*

---

## 🎮 Usage

Launch the primary Streamlit application:

```bash
streamlit run app2.py
```

### Dashboard Workflows:
1. **Single Image OCR:** Upload an image, toggle OpenCV preprocessing if it's heavily degraded, dial in your Anti-Hallucination confidence threshold limit, and execute the OCR.
2. **Ground Truth Evaluation:** In the sidebar, paste the exact expected string of a document. Execute the pipeline to automatically calculate the pipeline's exact CER metric against your upload.
3. **Batch Processing:** Switch to the Batch Evaluator tab, point it to a raw dataset directory path on your PC (e.g., `C:\User\Desktop\dataset_images`), and it will heavily iterate through the images, compiling an massive, unified `evaluation_results.json` dataset locally into the `batch_output` directory!

---
> *Architected and Developed for the DataThon TextTract OCR Computer Vision Track*

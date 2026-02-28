import os
import glob
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import uvicorn
import traceback

# Fix PaddlePaddle bug on Windows CPU
os.environ["FLAGS_enable_pir_api"] = "0"

app = FastAPI(title="PaddleOCR Web API")

# Allows Cross-Origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PaddleOCR
def load_ocr():
    return PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )

ocr = load_ocr()

# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/ocr")
async def run_ocr(file: UploadFile = File(...)):
    """Run OCR prediction and return the resulting image and JSON"""
    try:
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        # Save uploaded image
        # Use abs path to avoid working directory issues with paddlex pipeline
        abs_pwd = os.path.abspath(os.getcwd())
        image_path = os.path.join(abs_pwd, "uploads", file.filename)
        output_dir = os.path.join(abs_pwd, "output")

        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run PaddleOCR predict
        result = ocr.predict(image_path)

        # Save image + JSON for each result (Pipeline style)
        for res in result:
            res.save_to_img(output_dir)
            res.save_to_json(output_dir)

        # Find OCR result image and json
        img_files = sorted(
            glob.glob(os.path.join(output_dir, "*_ocr_res_img.png")) +
            glob.glob(os.path.join(output_dir, "*_ocr_res_img.jpg"))
        )
        json_files = sorted(glob.glob(os.path.join(output_dir, "*.json")))

        image_url = None
        json_url = None

        if img_files:
            image_url = f"/api/files/{os.path.basename(img_files[-1])}"
        if json_files:
            json_url = f"/api/files/{os.path.basename(json_files[-1])}"

        return JSONResponse(content={
            "success": True,
            "image_url": image_url,
            "json_url": json_url,
            "original_image_url": f"/api/uploads/{file.filename}"
        })
    except Exception as e:
        print("ERROR IN OCR:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "traceback": traceback.format_exc()})

@app.get("/api/files/{filename}")
async def get_output_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/api/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("web_backend:app", host="127.0.0.1", port=8000, reload=True)

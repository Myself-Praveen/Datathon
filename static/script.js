document.addEventListener("DOMContentLoaded", () => {

    // Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadIdle = document.getElementById("upload-idle");
    const uploadPreview = document.getElementById("upload-preview");
    const uploadLoading = document.getElementById("upload-loading");

    const imagePreview = document.getElementById("image-preview");
    const changeImageBtn = document.getElementById("change-image-btn");
    const runOcrBtn = document.getElementById("run-ocr-btn");

    const resultsSection = document.getElementById("results-section");
    const resultOriginal = document.getElementById("result-original");
    const resultAnnotated = document.getElementById("result-annotated");
    const downloadJsonUrl = document.getElementById("download-json-url");

    let selectedFile = null;

    // Trigger File Dialog
    dropZone.addEventListener("click", (e) => {
        if (uploadIdle.style.display !== "none") {
            fileInput.click();
        }
    });

    changeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // prevent clicking dropzone
        fileInput.click();
    });

    // Handle File Selection
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop Effects
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
            fileInput.files = e.dataTransfer.files; // sync with input
        }
    });

    function handleFile(file) {
        if (!file.type.match("image.*")) {
            alert("Please upload an image file (JPG, PNG).");
            return;
        }
        selectedFile = file;

        // Preview Image using ObjectUrl
        const objUrl = URL.createObjectURL(file);
        imagePreview.src = objUrl;

        // Show Preview State
        uploadIdle.style.display = "none";
        uploadPreview.style.display = "flex";
        resultsSection.style.display = "none"; // Hide old results
    }

    // Run OCR functionality
    runOcrBtn.addEventListener("click", async (e) => {
        e.stopPropagation(); // Prevent re-trigger file click

        if (!selectedFile) return;

        // Set Loading State
        uploadPreview.style.display = "none";
        uploadLoading.style.display = "flex";
        dropZone.style.cursor = "wait";

        // File upload using FormData
        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/api/ocr", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Server error.");

            const data = await response.json();

            if (data.success) {
                // Populate Result Grid
                resultOriginal.src = data.original_image_url;
                resultAnnotated.src = data.image_url;

                // Populate Download Link BaseURL handling
                downloadJsonUrl.href = data.json_url;

                // Show Final State
                uploadLoading.style.display = "none";
                uploadPreview.style.display = "flex";
                resultsSection.style.display = "block";

            } else {
                throw new Error("OCR execution failed.");
            }
        } catch (err) {
            alert("An error occurred during OCR prediction: " + err.message);
            // Reset state on failure
            uploadLoading.style.display = "none";
            uploadPreview.style.display = "flex";
        } finally {
            dropZone.style.cursor = "pointer";
        }

    });
});

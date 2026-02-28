import os
# Set to avoid PIR crash on Windows
os.environ["FLAGS_enable_pir_api"] = "0"
import glob
import json
import time
from paddleocr import PaddleOCR

def run_batch_inference():
    print("Loading PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    dataset_dir = r"C:\Users\prave\OneDrive\Desktop\test\data_subset\data_subset"
    output_dir = "batch_output"
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
    
    # Process only a subset so it doesn't take 5 hours (e.g. max 50 images)
    image_paths = image_paths[:50]
    
    results = {}
    
    print(f"Starting inference on {len(image_paths)} images...")
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"Processing [{i+1}/{len(image_paths)}]: {filename}")
        try:
            # Predict
            res = ocr.predict(img_path)
            
            # Parse result
            extracted_texts = []
            for item in res:
                # Based on the structure we used in app2.py
                try:
                    for text, score, poly in zip(item.rec_texts, item.rec_scores, item.dt_polys):
                        extracted_texts.append({
                            "text": text,
                            "confidence": score,
                            "box": [list(pt) for pt in poly]
                        })
                except Exception as ex:
                    # Fallback property lookup
                    try:
                        res_dict = item if isinstance(item, dict) else item.__dict__
                        if 'rec_texts' in res_dict and 'dt_polys' in res_dict:
                            for idx, text in enumerate(res_dict['rec_texts']):
                                extracted_texts.append({
                                    "text": text,
                                    "confidence": res_dict['rec_scores'][idx] if 'rec_scores' in res_dict else 1.0,
                                    "box": [list(pt) for pt in res_dict['dt_polys'][idx]]
                                })
                    except Exception as e:
                        pass

                        
            results[filename] = extracted_texts
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            
    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
    
    # Save results
    output_meta = os.path.join(output_dir, "evaluation_results.json")
    with open(output_meta, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_meta}")

if __name__ == "__main__":
    run_batch_inference()

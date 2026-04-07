# submission_task1.py
# Purpose: Task 1 inference script - exact competition format
# Reads test data from standard Kaggle path, generates short precise answers, outputs valid JSON
# Loads LoRA from HF nhattan9999t/UIT_NEWRON/task1/model (auto-downloaded)

import os
import json
import sys
import argparse
from huggingface_hub import snapshot_download
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
from PIL import Image
from preprocess import preprocess_image
import torch

# --------------------- CONFIG ---------------------
HF_REPO = "nhattan9999t/UIT_NEWRON"
MODEL_SUBFOLDER = "task1/model"
MAX_LENGTH = 32
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default="/kaggle/input/test/test_questions.json", help="Path to test questions JSON")
    parser.add_argument("--image_dir", default="/kaggle/input/test/images", help="Path to test images folder")
    parser.add_argument("--output", default="submission_task1.json", help="Output JSON path")
    args = parser.parse_args()

    # Download model from HF subfolder
    print("Downloading LoRA model from HF...")
    model_dir = snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{MODEL_SUBFOLDER}/**"])
    model_path = os.path.join(model_dir, MODEL_SUBFOLDER)
    
    processor = Blip2Processor.from_pretrained(model_path)
    base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-base", torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Load test data (assumed format: list of {"question_id": str, "image_id": str, "question": str})
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    results = []
    for item in test_data:
        img_path = os.path.join(args.image_dir, f"{item['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = preprocess_image(image)
        
        prompt = f"You are a medical expert. Question: {item['question']} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH, num_beams=3)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        # Normalize
        answer = answer.split("Answer:")[-1].strip().lower().replace(".", "").replace(",", "")
        
        results.append({
            "question_id": item["question_id"],
            "answer": answer
        })

    # Exact competition JSON format (list of predictions)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Task 1 submission saved to {args.output} - {len(results)} answers generated.")

if __name__ == "__main__":
    main()
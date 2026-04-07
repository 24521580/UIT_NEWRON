# submission_task2.py
# Purpose: Task 2 inference script with RAG + safety filter
# - Retrieves top-3 from knowledge base
# - Enhanced prompt with RAG context
# - Applies strict safety filter
# - Same output JSON format as Task 1

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
from rag_utils import retrieve_rag, safety_filter

# --------------------- CONFIG ---------------------
HF_REPO = "nhattan9999t/UIT_NEWRON"
MODEL_SUBFOLDER = "task2/model"
MAX_LENGTH = 32
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default="/kaggle/input/test/test_questions.json", help="Path to test questions JSON")
    parser.add_argument("--image_dir", default="/kaggle/input/test/images", help="Path to test images folder")
    parser.add_argument("--output", default="submission_task2.json", help="Output JSON path")
    args = parser.parse_args()

    # Download model from HF subfolder
    print("Downloading LoRA model from HF for Task 2...")
    model_dir = snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{MODEL_SUBFOLDER}/**"])
    model_path = os.path.join(model_dir, MODEL_SUBFOLDER)
    
    processor = Blip2Processor.from_pretrained(model_path)
    base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-base", torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    results = []
    for item in test_data:
        img_path = os.path.join(args.image_dir, f"{item['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = preprocess_image(image)
        
        # RAG context
        context = retrieve_rag(item["question"])
        
        prompt = f"Context: {context}\nYou are a medical expert. Question: {item['question']} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH, num_beams=3)
        raw_answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        # Extract answer part
        answer = raw_answer.split("Answer:")[-1].strip() if "Answer:" in raw_answer else raw_answer
        answer = safety_filter(answer)
        
        results.append({
            "question_id": item["question_id"],
            "answer": answer
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Task 2 submission saved to {args.output} - RAG + safety filter applied.")

if __name__ == "__main__":
    main()
# rag_utils.py
# Purpose: Task-2 RAG + safety filter
# - Loads competition-provided knowledge base (assumed JSON list of {"text": "..."} at /kaggle/input/knowledge-base/knowledge.json)
# - Embeds with all-MiniLM-L6-v2 + FAISS for top-k=3 retrieval
# - Injects into prompt
# - Safety filter: only "I am not certain" on genuine low-confidence (empty/short answer or uncertainty keywords)

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_EMB = SentenceTransformer("all-MiniLM-L6-v2")
KNOWLEDGE_PATH = "/kaggle/input/knowledge-base/knowledge.json"  # competition standard path

# Load KB once (cached)
if os.path.exists(KNOWLEDGE_PATH):
    with open(KNOWLEDGE_PATH, "r") as f:
        kb_data = json.load(f)
    passages = [item["text"] for item in kb_data]
    print(f"RAG: Loaded {len(passages)} knowledge passages")
else:
    passages = ["Endoscopy shows normal mucosa.", "Polyp detection requires biopsy confirmation.", "Ulcer indicates possible IBD."]  # fallback for dev
    print("RAG: Using fallback knowledge base")

embeddings = MODEL_EMB.encode(passages, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

def retrieve_rag(question: str, top_k: int = 3) -> str:
    """Retrieve top-k passages and format as context"""
    q_emb = MODEL_EMB.encode([question], normalize_embeddings=True)
    distances, indices = index.search(q_emb.astype(np.float32), top_k)
    retrieved = [passages[i] for i in indices[0]]
    return "\n".join(retrieved)

def safety_filter(answer: str) -> str:
    """Strict safety filter - never default to uncertain"""
    answer = answer.strip().lower()
    if not answer or len(answer) < 3 or any(word in answer for word in ["not sure", "uncertain", "unknown", "cannot determine"]):
        return "I am not certain"
    return answer.capitalize()  # normalized medical term style
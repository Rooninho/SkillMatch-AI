from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse

APP_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(APP_DIR), "data", "sample_jobs.csv")
MODEL_DIR = os.path.join(APP_DIR, "model_store")
VECT_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
MATRIX_PATH = os.path.join(MODEL_DIR, "job_matrix.npz")

app = FastAPI(title="AI Job Recommender", version="0.1.0")

class RecRequest(BaseModel):
    skills: str
    top_k: Optional[int] = 5

def _load_models():
    if not os.path.exists(VECT_PATH) or not os.path.exists(MATRIX_PATH):
        raise RuntimeError("Index not found. Run `python backend/build_index.py` first.")
    vectorizer = joblib.load(VECT_PATH)
    job_matrix = sparse.load_npz(MATRIX_PATH)
    jobs = pd.read_csv(DATA_PATH)
    return vectorizer, job_matrix, jobs

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecRequest):
    vectorizer, job_matrix, jobs = _load_models()
    query_vec = vectorizer.transform([req.skills])
    sims = cosine_similarity(query_vec, job_matrix).ravel()
    top_k = max(1, min(int(req.top_k or 5), len(sims)))
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        row = jobs.iloc[int(i)]
        results.append({
            "job_id": row["job_id"],
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "url": row["url"],
            "score": float(sims[int(i)]),
        })
    return {"query": req.skills, "results": results}
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later, replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later, replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later, replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

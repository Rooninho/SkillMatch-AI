# Vibe Hackathon 3.0 – AI Job Recommender (Starter Kit)

Build a simple AI that recommends jobs based on a person's skills. This starter kit gives you:
- A baseline recommender using TF–IDF + cosine similarity (no GPU needed)
- A FastAPI backend with `/recommend` and `/health` endpoints
- A tiny HTML frontend for quick demos
- Sample data and an index builder script
- Clear steps to upgrade to embeddings later

## Quickstart

### 1) Create & activate a virtual env (optional but recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Build the index (vectorizer + matrix) from sample jobs
```bash
python backend/build_index.py
```

### 4) Run the API
```bash
uvicorn backend.app:app --reload --port 8000
```

### 5) Open the demo UI
Just open `frontend/index.html` in your browser (or serve it with any static server).
Make sure the backend is running at `http://localhost:8000`.

---

## API

### `POST /recommend`
**Body:**
```json
{
  "skills": "python, data analysis, cloud aws, fastapi",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "python, data analysis, cloud aws, fastapi",
  "results": [
    {"job_id": "J1", "title": "Junior Data Analyst", "company": "Acme", "score": 0.71},
    ...
  ]
}
```

### `GET /health`
Returns `{"status":"ok"}` if the service is up.

---

## How it works (baseline)
- We vectorize job descriptions (title + required skills + summary) with TF–IDF
- We vectorize the user's skills text the same way
- We compute cosine similarity and return the top matches

### Upgrade path (embeddings)
- Replace TF–IDF with sentence embeddings (e.g., `sentence-transformers`) to improve matching
- Use a vector DB (FAISS, Chroma) for faster search on larger datasets
- Add re-ranking (e.g., hybrid BM25 + embeddings cosine)
- Use an LLM to extract structured skills from free-text CV/LinkedIn (optional)

---

## Repo structure
```
.
├── backend
│   ├── app.py
│   ├── build_index.py
│   └── model_store
│       ├── tfidf.pkl
│       └── job_matrix.npz
├── data
│   └── sample_jobs.csv
├── frontend
│   └── index.html
├── requirements.txt
└── README.md
```

> `model_store` is created by the index builder.

---


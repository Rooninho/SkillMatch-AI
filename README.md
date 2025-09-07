# Vibe Hackathon 3.0 – AI Job Recommender (Starter Kit)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-orange?style=flat-square)

**AI Job Recommender** is a simple Python-based project that recommends jobs based on a person's skills. This starter kit provides a baseline TF–IDF + cosine similarity recommender with a FastAPI backend and a minimal HTML frontend.

---

## 🚀 Features

- **TF–IDF + Cosine Similarity:** Baseline AI matching jobs to user skills.  
- **FastAPI Backend:** `/recommend` and `/health` endpoints for integration.  
- **Frontend Demo:** Tiny HTML page for quick testing.  
- **Sample Data & Index Builder:** Build job matrix with one command.  
- **Upgrade Path Ready:** Can replace TF–IDF with embeddings and vector databases.

---

## 🛠 Quickstart

### 1. Setup Virtual Environment (optional but recommended)

```bash
python -m venv .venv

1.Activate:
Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

3. Build the Index
python backend/build_index.py

4. Run the API
uvicorn backend.app:app --reload --port 8000

5. Open the Demo UI
Open frontend/index.html in your browser (or serve via any static server).

Ensure the backend is running at http://localhost:8000.

⚙## ⚙️ How It Works (Baseline)

1. Job descriptions (title + required skills + summary) are vectorized using **TF–IDF**.  
2. User skills are vectorized the same way.  
3. **Cosine similarity** is computed between user skills and jobs.  
4. Top K matches are returned.

---

## 🔮 Upgrade Path (Optional)

- Replace TF–IDF with **sentence embeddings** (e.g., `sentence-transformers`) for better semantic matching.  
- Use a **vector database** (FAISS, Chroma) for large datasets.  
- Add **re-ranking** (hybrid BM25 + embeddings cosine).  
- Integrate an **LLM** to extract structured skills from free-text CVs or LinkedIn profiles.

---

## 📁 Repository Structure


├── backend
│ ├── app.py
│ ├── build_index.py
│ └── model_store
│ ├── tfidf.pkl
│ └── job_matrix.npz
├── data
│ └── sample_jobs.csv
├── frontend
│ └── index.html
├── requirements.txt
└── README.md









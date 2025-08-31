import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

APP_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(APP_DIR), "data", "sample_jobs.csv")
MODEL_DIR = os.path.join(APP_DIR, "model_store")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_jobs():
    df = pd.read_csv(DATA_PATH)
    # combine text fields for vectorization
    df["text"] = df["title"].fillna("") + " | " + df["skills"].fillna("") + " | " + df["summary"].fillna("")
    return df

def main():
    df = load_jobs()
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    X = vectorizer.fit_transform(df["text"].tolist())
    # persist
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.pkl"))
    sparse.save_npz(os.path.join(MODEL_DIR, "job_matrix.npz"), X)
    print(f"Built index: {X.shape[0]} jobs, {X.shape[1]} features")

if __name__ == "__main__":
    main()

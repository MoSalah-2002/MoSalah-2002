"""Arabic Hadith semantic search using GATE-AraBERT embeddings."""
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
DATA_PATH = "data/kaggle_hadiths_clean.csv"

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\x60“’”‘ﷺ]", "", text)
    text = text.replace("\u200f", "").strip()
    text = re.sub(r"""[\x60"' ]+""", " ", text)
    return re.sub(r"\s+", " ", text).strip()

class HadithSearch:
    def __init__(self, texts, model_name=MODEL_NAME, device=None):
        self.texts = list(texts)
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = self.model.encode(
            self.texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True, batch_size=32,
        )

    def search(self, query, top_k=5):
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.embeddings @ query_vec
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.texts[i], "score": float(scores[i])}
            for i in indices
        ]

def main():
    df = pd.read_csv(DATA_PATH)
    texts = df["text_ar"].map(preprocess_text)
    texts = texts[texts.str.len() > 0].tolist()
    searcher = HadithSearch(texts)
    while True:
        query = input("Arabic query (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        for rank, item in enumerate(searcher.search(query, top_k=5), 1):
            print(f"\n{rank}. score={item['score']:.4f}\n{item['text']}")

if __name__ == "__main__":
    main()

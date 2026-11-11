# Arabic Hadith Semantic Search

A semantic retrieval experiment for finding Arabic Hadith passages related to a natural-language query.

## Method
1. Load the Arabic Hadith corpus from the referenced Kaggle dataset.
2. Normalize whitespace and remove selected punctuation/formatting characters.
3. Encode Hadith texts using `Omartificial-Intelligence-Space/GATE-AraBert-v1`.
4. Rank passages by cosine similarity to the query embedding.
5. Return the top-k closest passages.

## Dataset
The original notebook downloads `alizahidraja/quran-nlp` from Kaggle and reads `data/hadith/kaggle_hadiths_clean.csv`. Dataset files are not included here.

## Run
Install `requirements.txt`, download and extract the dataset, then open `notebooks/hadith-semantic-search.ipynb`. The original experiment encodes the corpus on GPU; CPU is possible but slower.

## Limitations
Similarity is not proof of authenticity, correctness, or a legal/religious ruling. Retrieval quality has not been evaluated with a labelled benchmark, so results should be treated as candidate passages for human review. The notebook's original per-query loop is a simple baseline and may be slow for a large corpus.

## Stack
Python, Sentence Transformers, GATE-AraBERT, scikit-learn, NumPy, Pandas.

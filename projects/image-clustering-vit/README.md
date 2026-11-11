# Image Clustering with Vision Transformers

An unsupervised image-clustering experiment using pretrained Vision Transformer (ViT) embeddings and scikit-learn clustering algorithms.

## Pipeline
1. Load images from a local directory.
2. Extract image-level embeddings using the CLS token from `google/vit-base-patch16-224`.
3. L2-normalize embeddings.
4. Cluster images with K-Means or Agglomerative Clustering.
5. Inspect cluster assignments and evaluate them with the silhouette score when appropriate.

## Requirements
Python 3.10+; install dependencies:

```bash
pip install -r requirements.txt
```

## Data
Create an `images/` directory and add your own JPG, JPEG, or PNG images. Images are not included in this repository.

## Run

```bash
python src/cluster_images.py --image-dir images --clusters 5 --method kmeans
```

To use agglomerative clustering:

```bash
python src/cluster_images.py --image-dir images --clusters 5 --method agglomerative
```

The script writes `cluster_assignments.csv` containing each filename and its assigned cluster.

## Notes and limitations
- This is an exploratory unsupervised-learning project, not a trained image classifier.
- Cluster quality depends on the images, chosen number of clusters, and embedding model.
- Cluster IDs are arbitrary and do not represent semantic class names.
- Silhouette score is only a geometric diagnostic; it does not establish semantic correctness.
- The original experiment used a fixed value of 10 clusters and did not include a labelled benchmark. This cleaned version makes the number of clusters configurable.

## Model
`google/vit-base-patch16-224` (pretrained feature extractor; no fine-tuning).

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, ViTModel


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def load_embeddings(image_dir: Path, model_name: str):
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not paths:
        raise ValueError(f"No supported images found in {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device).eval()

    features, valid_paths = [], []
    with torch.inference_mode():
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                print(f"Skipping {path.name}: {exc}")
                continue
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # CLS token is used as a fixed-size image representation.
            vector = outputs.last_hidden_state[:, 0, :].squeeze(0)
            features.append(vector.cpu().numpy())
            valid_paths.append(path)

    if len(features) < 2:
        raise ValueError("At least two readable images are required.")
    return valid_paths, normalize(np.asarray(features))


def main():
    parser = argparse.ArgumentParser(description="Cluster images using pretrained ViT embeddings.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--method", choices=["kmeans", "agglomerative"], default="kmeans")
    parser.add_argument("--model", default="google/vit-base-patch16-224")
    parser.add_argument("--output", type=Path, default=Path("cluster_assignments.csv"))
    args = parser.parse_args()

    paths, embeddings = load_embeddings(args.image_dir, args.model)
    if not 2 <= args.clusters <= len(paths):
        raise ValueError(f"--clusters must be between 2 and {len(paths)}")

    if args.method == "kmeans":
        estimator = KMeans(n_clusters=args.clusters, random_state=42, n_init="auto")
    else:
        estimator = AgglomerativeClustering(n_clusters=args.clusters, metric="cosine", linkage="average")

    labels = estimator.fit_predict(embeddings)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "cluster"])
        writer.writerows((path.name, int(label)) for path, label in zip(paths, labels))

    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        print(f"Silhouette score (cosine): {silhouette_score(embeddings, labels, metric='cosine'):.4f}")
    print(f"Saved {len(paths)} assignments to {args.output}")


if __name__ == "__main__":
    main()

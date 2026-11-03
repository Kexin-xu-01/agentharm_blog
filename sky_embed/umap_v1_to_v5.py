# Joint UMAP of the v1, v2.1, v3, v4 and v5 prompt embeddings (same settings as the v1-only figure).
# Row order of the output: v1, v2.1 (texts_v2.1.jsonl order), v3, v4, v5.
import numpy as np, umap
parts = ["", "_v2.1", "_v3", "_v4", "_v5"]
X = np.vstack([np.load(f"emb_all-mpnet-base-v2{p}.npy") for p in parts])
Y = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=42).fit_transform(X)
np.save("umap_semantic_v1_to_v5_s42.npy", Y)
print("saved", Y.shape)

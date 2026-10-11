# Joint UMAP of the v1 and v2.1 prompt embeddings (same settings as the v1-only figure).
import numpy as np, umap
A = np.load("emb_all-mpnet-base-v2.npy"); B = np.load("emb_all-mpnet-base-v2_v2.1.npy")
Y = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=42).fit_transform(np.vstack([A, B]))
np.save("umap_semantic_v1_v2.1_s42.npy", Y)
print("saved", Y.shape)

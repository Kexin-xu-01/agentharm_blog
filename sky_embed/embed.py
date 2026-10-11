import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer
model = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else "texts.jsonl"  # texts_<tag>.jsonl -> emb_<model>_<tag>.npy
tag = src[len("texts"):-len(".jsonl")]
texts = [json.loads(l) for l in open(src)]
X = SentenceTransformer(model).encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
np.save(f"emb_{model.replace('/', '_')}{tag}.npy", X.astype(np.float32))
print("saved", X.shape)

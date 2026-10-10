import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer
model = sys.argv[1]
texts = [json.loads(l) for l in open("texts.jsonl")]
X = SentenceTransformer(model).encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
np.save(f"emb_{model.replace('/', '_')}.npy", X.astype(np.float32))
print("saved", X.shape)

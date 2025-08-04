import faiss
from model.distiluse_embedder import encode

def build_faiss_index(corpus):
    vectors = encode(corpus)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

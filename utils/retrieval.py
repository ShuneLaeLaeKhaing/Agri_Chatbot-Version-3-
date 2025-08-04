from utils.preprocessing import tokenize_myanmar
from model.distiluse_embedder import encode

def retrieve(query, bm25, faiss, corpus, answers):
    # BM25 Retrieval
    tokens = tokenize_myanmar(query)
    bm_scores = bm25.get_scores(tokens)
    bm_idx = bm_scores.argmax()
    bm_answer = answers[bm_idx]

    # FAISS Retrieval
    vec = encode([query])
    _, faiss_idx = faiss.search(vec, k=1)
    faiss_answer = answers[faiss_idx[0][0]]

    # Combine
    return bm_answer if bm_scores.max() > 1.3 else faiss_answer

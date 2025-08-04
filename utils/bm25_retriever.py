from rank_bm25 import BM25Okapi
from utils.preprocessing import tokenize_myanmar

def build_bm25(corpus):
    tokenized_corpus = [tokenize_myanmar(text) for text in corpus]
    return BM25Okapi(tokenized_corpus)

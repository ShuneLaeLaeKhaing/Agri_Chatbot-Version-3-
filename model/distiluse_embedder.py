from sentence_transformers import SentenceTransformer

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

def encode(texts):
    return model.encode(texts, convert_to_numpy=True)



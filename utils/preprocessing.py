import utils.sylbreak as sylbreak
from utils.sylbreak import syllable_tokenizer
import json

def tokenize_myanmar(text):
    return syllable_tokenizer(text)

def load_agri_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    corpus, answers, meta = [], [], []
    for entry in data:
        questions = [entry["question_mm"]] + entry.get("paraphrase_mm", [])
        for q in questions:
            corpus.append(q)
            answers.append(entry["answer_mm"])
            meta.append(entry)
    return corpus, answers, meta





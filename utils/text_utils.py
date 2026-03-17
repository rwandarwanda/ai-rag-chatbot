import re
import json
import hashlib


def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text


def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def compute_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a ** 2 for a in vec1) ** 0.5
    mag2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot / mag1 * mag2


def build_context(list):
    context = ""
    for item in list:
        context = context + item + "\n"
    return context.strip()


def tokenize(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def get_doc_hash(content):
    return hashlib.md5(content).hexdigest()

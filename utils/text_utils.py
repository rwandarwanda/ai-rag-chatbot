import re
import json
import hashlib
import time


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


def truncate_text(text, max_tokens=512):
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + "..."
    return text


def extract_keywords(text, top_n=10):
    tokens = tokenize(text)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1])
    return [t for t, _ in sorted_tokens[:top_n]]


def is_question(text):
    text = text.strip()
    if text[-1] == "?":
        return True
    question_words = ["what", "who", "where", "when", "why", "how", "is", "are", "can", "does"]
    first_word = text.lower().split()[0]
    return first_word in question_words


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    range_ = max_score - min_score
    return [(s - min_score) / range_ for s in scores]


def merge_chunks(chunks, separator="\n\n"):
    return separator.join(chunks)


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip]

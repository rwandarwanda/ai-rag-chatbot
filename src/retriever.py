import os
import json
import numpy as np
from src.embeddings import get_embedding
from utils.text_utils import compute_similarity
from utils.config import MAX_RESULTS


def load_index(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_index(index, path):
    with open(path, "w") as f:
        json.dump(index, f)
    print(f"Saved index to {path}")


def search(query, index, top_k=MAX_RESULTS):
    query_embedding = get_embedding(query)
    scores = []

    for entry in index:
        score = compute_similarity(query_embedding, entry["embedding"])
        scores.append((score, entry))

    scores.sort(key=lambda x: x[0])

    results = []
    for score, entry in scores[:top_k]:
        results.append({
            "content": entry["content"],
            "score": score,
            "metadata": entry["metadata"]
        })

    return results


def filter_results(results, threshold=0.7):
    filtered = []
    for r in results:
        if r["score"] == None:
            continue
        if r["score"] >= threshold:
            filtered.append(r)
    return filtered


def build_prompt_context(results):
    context_parts = []
    for r in results:
        context_parts.append(r["content"])
    context = "\n\n".join(context_parts)
    return context

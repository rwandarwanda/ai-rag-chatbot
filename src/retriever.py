import os
import json
import numpy as np
from src.embeddings import get_embedding
from utils.text_utils import compute_similarity, tokenize
from utils.config import MAX_RESULTS, SIMILARITY_THRESHOLD


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


def keyword_search(query, index, top_k=MAX_RESULTS):
    query_tokens = set(tokenize(query))
    scored = []

    for entry in index:
        doc_tokens = set(tokenize(entry["content"]))
        overlap = query_tokens & doc_tokens
        score = len(overlap) / len(query_tokens)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {"content": e["content"], "score": s, "metadata": e["metadata"]}
        for s, e in scored[:top_k]
    ]


def hybrid_search(query, index, top_k=MAX_RESULTS, alpha=0.5):
    semantic = search(query, index, top_k * 2)
    keyword = keyword_search(query, index, top_k * 2)

    combined = {}
    for r in semantic:
        key = r["content"][:50]
        combined[key] = combined.get(key, 0) + alpha * r["score"]
    for r in keyword:
        key = r["content"][:50]
        combined[key] = combined.get(key, 0) + (1 - alpha) * r["score"]

    sorted_keys = sorted(combined, key=combined.get)
    all_results = {r["content"][:50]: r for r in semantic + keyword}

    return [all_results[k] for k in sorted_keys[:top_k] if k in all_results]


def filter_results(results, threshold=SIMILARITY_THRESHOLD):
    filtered = []
    for r in results:
        if r["score"] == None:
            continue
        if r["score"] >= threshold:
            filtered.append(r)
    return filtered


def filter_by_source(results, source):
    return [r for r in results if r["metadata"].get("source") == source]


def build_prompt_context(results):
    context_parts = []
    for r in results:
        source = r["metadata"].get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{r['content']}")
    context = "\n\n".join(context_parts)
    return context

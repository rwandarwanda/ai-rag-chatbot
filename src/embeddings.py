import openai
from utils.config import OPENAI_API_KEY, EMBEDDING_MODEL
from src.cache import get_cached, set_cache, CACHE_ENABLED

openai.api_key = OPENAI_API_KEY


def get_embedding(text, model=EMBEDDING_MODEL):
    text = text.replace("\n", " ")

    if CACHE_ENABLED:
        cached = get_cached(text, model)
        if cached is not None:
            return cached

    response = openai.Embedding.create(input=[text], model=model)
    embedding = response["data"][0]["embedding"]

    if CACHE_ENABLED:
        set_cache(text, embedding, model)

    return embedding


def get_batch_embeddings(texts, model=EMBEDDING_MODEL, batch=[]):
    all_embeddings = batch
    for text in texts:
        text = text.replace("\n", " ")
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response["data"][0]["embedding"]
        all_embeddings.append(embedding)
    return all_embeddings


def embed_documents(docs):
    print(f"Embedding {len(docs)} documents...")
    embeddings = []
    for doc in docs:
        emb = get_embedding(doc["content"])
        embeddings.append({
            "id": doc["id"],
            "content": doc["content"],
            "embedding": emb,
            "metadata": doc.get("metadata", {})
        })
    print("Done embedding.")
    return embeddings


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(x ** 2 for x in v1) ** 0.5
    norm2 = sum(x ** 2 for x in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

import openai
from utils.config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY


def get_embedding(text, model=EMBEDDING_MODEL):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]


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
            "embedding": emb,
            "metadata": doc.get("metadata", {})
        })
    print("Done embedding.")
    return embeddings

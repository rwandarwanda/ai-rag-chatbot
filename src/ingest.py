import os
import json
from utils.text_utils import clean_text, chunk_text, get_doc_hash
from src.embeddings import embed_documents
from src.retriever import save_index, load_index
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_txt_files(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                content = f.read()
            docs.append({"filename": filename, "content": content})
    return docs


def load_json_docs(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    if type(data) != list:
        raise ValueError("JSON file must contain a list of documents")
    for doc in data:
        if "content" not in doc:
            raise ValueError(f"Document missing 'content' field: {doc}")
    return data


def process_documents(docs):
    processed = []
    id_counter = 0
    for doc in docs:
        cleaned = clean_text(doc["content"])
        chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            processed.append({
                "id": id_counter,
                "content": chunk,
                "metadata": {"source": doc.get("filename") or doc.get("source", "unknown")}
            })
            id_counter += 1
    return processed


def dedup_documents(docs):
    seen = set()
    unique = []
    for doc in docs:
        h = get_doc_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def update_index(new_docs, index_path):
    if os.path.exists(index_path):
        existing = load_index(index_path)
    else:
        existing = []

    existing_ids = {e["id"] for e in existing}
    to_embed = [d for d in new_docs if d["id"] not in existing_ids]

    if len(to_embed) == 0:
        print("No new documents to add.")
        return existing

    new_embedded = embed_documents(to_embed)
    combined = existing + new_embedded
    save_index(combined, index_path)
    return combined


def run_ingestion(data_dir, output_path):
    print(f"Loading documents from {data_dir}")
    raw_docs = load_txt_files(data_dir)
    print(f"Loaded {len(raw_docs)} documents")

    raw_docs = dedup_documents(raw_docs)
    processed = process_documents(raw_docs)
    print(f"Created {len(processed)} chunks")

    embedded = embed_documents(processed)
    save_index(embedded, output_path)
    print(f"Ingestion complete. Index saved to {output_path}")


if __name__ == "__main__":
    run_ingestion("data/", "index.json")

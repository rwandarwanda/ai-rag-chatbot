import os
from utils.text_utils import clean_text, chunk_text, get_doc_hash
from src.embeddings import embed_documents
from src.retriever import save_index
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
                "metadata": {"source": doc["filename"]}
            })
            id_counter += 1
    return processed


def run_ingestion(data_dir, output_path):
    print(f"Loading documents from {data_dir}")
    raw_docs = load_txt_files(data_dir)
    print(f"Loaded {len(raw_docs)} documents")

    processed = process_documents(raw_docs)
    print(f"Created {len(processed)} chunks")

    embedded = embed_documents(processed)
    save_index(embedded, output_path)
    print(f"Ingestion complete. Index saved to {output_path}")


if __name__ == "__main__":
    run_ingestion("data/", "index.json")

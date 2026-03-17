# ai-rag-chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot powered by OpenAI.

## Features

- Document ingestion from `.txt` and `.json` files
- Semantic search using OpenAI embeddings
- Hybrid search (semantic + keyword)
- LLM reranking for improved retrieval precision
- Response caching to reduce API costs
- Streaming response support
- Conversation history with context window management

## Project Structure

```
ai-rag-chatbot/
├── main.py               # Entry point
├── requirements.txt
├── data/                 # Place your source documents here
├── src/
│   ├── chatbot.py        # Chat loop and response generation
│   ├── embeddings.py     # Embedding generation and caching
│   ├── ingest.py         # Document loading and processing
│   ├── retriever.py      # Search and retrieval logic
│   ├── reranker.py       # LLM-based reranking
│   └── cache.py          # In-memory response cache
└── utils/
    ├── config.py         # All configuration constants
    └── text_utils.py     # Text processing utilities
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your API keys in `utils/config.py` (or via environment variables).

3. Add your documents to the `data/` directory.

4. Run ingestion to build the index:
   ```bash
   python -m src.ingest
   ```

5. Start the chatbot:
   ```bash
   python main.py index.json
   ```

## Configuration

All settings live in `utils/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_MODEL` | `gpt-4` | Model used for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Model used for embeddings |
| `CHUNK_SIZE` | `512` | Words per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MAX_RESULTS` | `5` | Number of documents retrieved |
| `RERANK_TOP_K` | `3` | Results kept after reranking |
| `CACHE_TTL` | `3600` | Cache entry TTL in seconds |
| `SIMILARITY_THRESHOLD` | `0.72` | Minimum score to include a result |

## Usage

```python
from src.retriever import load_index
from src.chatbot import ask

index = load_index("index.json")
answer = ask("What is retrieval-augmented generation?", index)
print(answer)
```

import sys
from src.chatbot import run_chat_loop
from src.retriever import load_index


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <index_path>")
        sys.exit(1)

    index_path = sys.argv[1]
    index = load_index(index_path)
    run_chat_loop(index)


if __name__ == "__main__":
    main()

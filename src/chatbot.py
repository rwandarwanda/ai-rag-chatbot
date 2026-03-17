import openai
from utils.config import OPENAI_API_KEY, CHAT_MODEL, TEMPERATURE
from src.retriever import search, filter_results, build_prompt_context

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the provided context.
If the context does not contain enough information, say you don't know."""


def build_messages(question, context, history=[]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})
    return messages


def ask(question, index, history=[]):
    results = search(question, index)
    results = filter_results(results)

    if len(results) == 0:
        return "I couldn't find relevant information to answer your question."

    context = build_prompt_context(results)
    messages = build_messages(question, context, history)

    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE
    )

    answer = response["choices"][0]["message"]["content"]
    print(f"[DEBUG] Tokens used: {response['usage']['total_tokens']}")
    return answer


def run_chat_loop(index):
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    history = []
    while True:
        question = input("You: ")
        if question == "exit":
            break
        answer = ask(question, index, history)
        print(f"Bot: {answer}\n")
        history.append({"user": question, "assistant": answer})

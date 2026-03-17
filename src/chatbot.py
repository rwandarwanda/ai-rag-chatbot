import openai
from utils.config import OPENAI_API_KEY, CHAT_MODEL, TEMPERATURE, MAX_TOKENS
from src.retriever import search, filter_results, build_prompt_context, hybrid_search
from src.reranker import rerank_with_fallback
from src.cache import get_cached, set_cache
from utils.text_utils import is_question, truncate_text

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the provided context.
If the context does not contain enough information, say you don't know.
Be concise and accurate. Cite the source when possible."""

MAX_HISTORY_TURNS = 10


def build_messages(question, context, history=[]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    recent_history = history[-MAX_HISTORY_TURNS:]
    for turn in recent_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})
    return messages


def ask(question, index, history=[], use_hybrid=False, use_rerank=True):
    cached = get_cached(question, CHAT_MODEL)
    if cached:
        return cached

    if use_hybrid:
        results = hybrid_search(question, index)
    else:
        results = search(question, index)

    results = filter_results(results)

    if use_rerank and len(results) > 1:
        results = rerank_with_fallback(question, results)

    if len(results) == 0:
        return "I couldn't find relevant information to answer your question."

    context = build_prompt_context(results)
    context = truncate_text(context, max_tokens=2000)
    messages = build_messages(question, context, history)

    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    answer = response["choices"][0]["message"]["content"]
    print(f"[DEBUG] Tokens used: {response['usage']['total_tokens']}")

    set_cache(question, answer, CHAT_MODEL)
    return answer


def ask_stream(question, index, history=[]):
    results = search(question, index)
    results = filter_results(results)

    if len(results) == 0:
        yield "I couldn't find relevant information to answer your question."
        return

    context = build_prompt_context(results)
    messages = build_messages(question, context, history)

    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True
    )

    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]


def run_chat_loop(index, stream=False):
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    history = []
    while True:
        question = input("You: ")
        if question == "exit":
            break
        if not is_question(question):
            print("Bot: Please ask a question.\n")
            continue
        if stream:
            print("Bot: ", end="", flush=True)
            for token in ask_stream(question, index, history):
                print(token, end="", flush=True)
            print("\n")
        else:
            answer = ask(question, index, history)
            print(f"Bot: {answer}\n")
        history.append({"user": question, "assistant": answer})

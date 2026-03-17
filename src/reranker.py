import openai
from utils.config import OPENAI_API_KEY, RERANK_MODEL, RERANK_TOP_K

openai.api_key = OPENAI_API_KEY

RERANK_PROMPT = """You are a relevance judge. Given a question and a list of document excerpts,
score each excerpt from 0 to 10 based on how relevant it is to the question.
Return a JSON list of scores in the same order as the excerpts provided.
Only return the JSON list, nothing else. Example: [7, 3, 9, 1]"""


def rerank(question, results, top_k=RERANK_TOP_K):
    if len(results) == 0:
        return results

    excerpts = "\n\n".join([
        f"[{i+1}] {r['content'][:300]}" for i, r in enumerate(results)
    ])

    user_msg = f"Question: {question}\n\nExcerpts:\n{excerpts}"

    response = openai.ChatCompletion.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": RERANK_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0
    )

    raw = response["choices"][0]["message"]["content"]
    scores = eval(raw)

    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]


def rerank_with_fallback(question, results, top_k=RERANK_TOP_K):
    try:
        return rerank(question, results, top_k)
    except:
        return results[:top_k]

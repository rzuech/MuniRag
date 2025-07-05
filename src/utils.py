def build_prompt(context_docs, user_q):
    joined = "\n\n".join(
        f"[Source {i+1}] {doc}" for i, (doc, _) in enumerate(context_docs)
    )
    return (
        "You are an AI assistant for municipal documents.\n"
        "Answer ONLY from the provided sources. If unsure, say you don't know.\n\n"
        f"{joined}\n\nQuestion: {user_q}\n\nAnswer (cite like [Source 1]):"
    )

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.memory_manager import get_memory

SYSTEM_PROMPT = (
    "You are a helpful assistant currently running as {model_name} via {provider}. "
    "Be transparent when you are uncertain. Do not fabricate citations or facts. "
    "If you do not know something, say so clearly."
)

RAG_GROUNDED_PROMPT = (
    "You are a helpful assistant currently running as {model_name} via {provider}. "
    "Answer using ONLY the retrieved document excerpts below. "
    "You may synthesise and infer from the excerpts — you do not need a verbatim match. "
    "Do not introduce any facts, claims, or context from your training data. "
    "If the excerpts contain no relevant information at all, say: "
    "'The retrieved sections do not contain enough information to answer this.' "
    "Cite the page number when relevant.\n\n"
    "Retrieved context:\n{context}"
)

RAG_AUGMENTED_PROMPT = (
    "You are a helpful assistant currently running as {model_name} via {provider}. "
    "You have been given the most relevant sections retrieved from a document. "
    "These excerpts may not cover every part of the document — for broad questions such as "
    "'summarize the document', work with what is provided and clearly state that your answer "
    "is based on the retrieved sections, not the full document. "
    "You may supplement retrieved context with your parametric knowledge, but clearly "
    "distinguish document-sourced claims from your own knowledge. "
    "Cite the page number when relevant.\n\n"
    "Retrieved context:\n{context}"
)


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page + 1}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(model: BaseChatModel, model_name: str, provider: str, session_id: str):
    """Plain conversational chain with window memory."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(model_name=model_name, provider=provider)),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    memory = get_memory(session_id)

    def load_history(_: dict) -> list:
        return memory.load_memory_variables({})["history"]

    chain = (
        RunnablePassthrough.assign(history=RunnableLambda(load_history))
        | prompt
        | model
    )
    return chain, memory


def build_rag_chain(model: BaseChatModel, model_name: str, provider: str,
                    session_id: str, retriever, grounded: bool = True):
    """RAG chain: retrieves context chunks before calling the LLM.

    grounded=True  → strict retrieval-only (no parametric knowledge)
    grounded=False → retrieval + parametric knowledge blending
    """
    memory = get_memory(session_id)

    def load_history(_: dict) -> list:
        return memory.load_memory_variables({})["history"]

    def retrieve_context(inputs: dict) -> str:
        docs = retriever.invoke(inputs["input"])
        return _format_docs(docs)

    template = RAG_GROUNDED_PROMPT if grounded else RAG_AUGMENTED_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", template.format(
            model_name=model_name, provider=provider, context="{context}"
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(load_history),
            context=RunnableLambda(retrieve_context),
        )
        | prompt
        | model
    )
    return chain, memory

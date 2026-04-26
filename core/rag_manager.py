from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.embedding_factory import get_embedding_model

_stores: dict[str, InMemoryVectorStore] = {}
_chunks: dict[str, list[Document]] = {}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)


def index_pdf(session_id: str, pdf_path: str) -> int:
    """Load, chunk, embed and store a PDF. Returns number of chunks indexed."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunks = _splitter.split_documents(pages)
    embeddings = get_embedding_model()
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    _stores[session_id] = store
    _chunks[session_id] = chunks
    return len(chunks)


def get_retriever(session_id: str, k: int = 8):
    """Return a retriever for this session, or None if no PDF is indexed."""
    store = _stores.get(session_id)
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": k})


def get_chunks(session_id: str) -> list[Document]:
    return _chunks.get(session_id, [])


def has_document(session_id: str) -> bool:
    return session_id in _stores


def test_retrieval(session_id: str, query: str, k: int = 4) -> list[dict]:
    """Return top-k chunks for query with similarity scores. Empty list if no doc."""
    store = _stores.get(session_id)
    if store is None:
        return []
    results = store.similarity_search_with_score(query, k=k)
    output = []
    for doc, score in results:
        output.append({
            "score": round(float(score), 4),
            "page": doc.metadata.get("page", 0) + 1,
            "text": doc.page_content,
        })
    return output


def clear_document(session_id: str) -> None:
    _stores.pop(session_id, None)
    _chunks.pop(session_id, None)

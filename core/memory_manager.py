import uuid
from langchain_classic.memory import ConversationBufferWindowMemory

from config.settings import MEMORY_WINDOW_SIZE

_sessions: dict[str, ConversationBufferWindowMemory] = {}


def new_session_id() -> str:
    return str(uuid.uuid4())


def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            return_messages=True,
            memory_key="history",
        )
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    if session_id in _sessions:
        _sessions[session_id].clear()

"""
Per-chat RAG. Uploaded PDFs/TXT files get chunked and embedded into a
Chroma collection scoped to that chat's thread_id, so the retrieve_context
tool (tools.py) can pull back just the relevant passages for a query -
instead of the old approach (app.py's get_file_content + graph.py's
prepare_input) of dumping the *entire* extracted file text into the very
next message, which only ever helped for one turn and could blow past the
model's context window on a large document.

Storage layout: one Chroma collection per thread_id, persisted under
./chroma_db/<thread_id>/ - kept alongside chatbot_checkpoints.sqlite so a
chat's uploaded documents survive a restart the same way its message
history does. delete_thread_store() is called from app.py's delete_chat()
so a deleted chat doesn't leave orphaned embeddings behind.

Uses the OpenAI embeddings client (text-embedding-3-small - cheap, and
OPENAI_API_KEY is already required for image generation and TTS, so this
doesn't add a new key requirement). Embeddings, like image gen and TTS,
always go through OpenAI regardless of which `source` you're chatting
with, same reasoning as multimodal.py.
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PERSIST_ROOT = Path("chroma_db")

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)


def _store_for(thread_id: str) -> Chroma:
    return Chroma(
        collection_name=f"chat_{thread_id}",
        embedding_function=_embeddings,
        persist_directory=str(_PERSIST_ROOT / str(thread_id)),
    )


def ingest_text(thread_id: str, filename: str, text: str) -> int:
    """Chunks and embeds `text`, tagging every chunk with its source
    filename so retrieve() results can say which document they came from.
    Returns the number of chunks written (used for the upload-status
    message in app.py). No-ops (returns 0) on empty/unextractable text."""
    if not text.strip():
        return 0
    chunks = _splitter.split_text(text)
    store = _store_for(thread_id)
    store.add_texts(
        texts=chunks,
        metadatas=[{"source": filename} for _ in chunks],
    )
    return len(chunks)


def retrieve(thread_id: str, query: str, k: int = 4) -> str:
    """Top-k matching chunks for this chat's uploaded documents, formatted
    as a single string for a ToolMessage. If nothing's been uploaded to
    this chat yet, this still round-trips normally - it just tells the
    model there's nothing to find, rather than erroring."""
    if not query:
        return "No query provided."
    store = _store_for(thread_id)
    docs = store.similarity_search(query, k=k)
    if not docs:
        return "No relevant content found in uploaded documents for this chat."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs
    )


def delete_thread_store(thread_id: str) -> None:
    """Removes a chat's embedded documents entirely. Mirrors what
    checkpointer.delete_thread() already does for the sqlite-backed
    message history, so deleting a chat cleans up both stores."""
    shutil.rmtree(_PERSIST_ROOT / str(thread_id), ignore_errors=True)

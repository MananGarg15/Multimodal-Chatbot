"""
Persists the sidebar chat list (stable id + display name) to a local JSON
file, separate from the SqliteSaver checkpoint file that holds actual
message history (chatbot_checkpoints.sqlite).

Why this exists: chat_list = gr.State(["Chat1"]) was only ever an
in-memory default. Restarting the app reset that Python state, so the
sidebar "forgot" every chat even though their message histories were
still sitting safely in the sqlite checkpoint file the whole time. This
module is what the sidebar reads on startup and writes to on every
create/rename/delete, so the two stores stay in sync.

Ids are stable and independent of list position (previously chat_no was
just i + 1 in the render loop). Position-based ids break the moment you
delete a chat in the middle of the list - every chat after it silently
shifts to a new "id", so a rename or delete issued right after a delete
can land on the wrong thread_id. A monotonically-increasing stable id
avoids that entirely: an id, once assigned, never changes for the
lifetime of that chat.
"""

import json
from pathlib import Path

_STORE_PATH = Path("chat_list.json")

_DEFAULT = [{"id": "1", "name": "Chat1"}]


def load_chats() -> list[dict]:
    """Returns [{"id": "...", "name": "..."}, ...], most-recently-created
    last. Creates the store with a single default chat on first run."""
    if not _STORE_PATH.exists():
        save_chats(_DEFAULT)
        return [dict(c) for c in _DEFAULT]

    try:
        with open(_STORE_PATH, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable store - don't crash the app over a sidebar
        # list, fall back to the default and let the user rebuild it.
        data = None

    if not data:
        data = [dict(c) for c in _DEFAULT]
        save_chats(data)
    return data


def save_chats(chat_list: list[dict]) -> None:
    with open(_STORE_PATH, "w") as f:
        json.dump(chat_list, f, indent=2)


def next_chat_id(chat_list: list[dict]) -> str:
    """Next stable id. NOT len(chat_list) + 1 - that collides after a
    delete, e.g. [id=1, id=2, id=3], delete id=2 -> len is 2, so len+1
    would hand out id=3 again, colliding with the surviving chat."""
    existing = [int(c["id"]) for c in chat_list if str(c["id"]).isdigit()]
    return str(max(existing, default=0) + 1)

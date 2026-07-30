"""
Shared graph state for the combined multimodal / multi-chat bot.

Replaces:
- Day4LLMCalling/messageSeries.py's mSeries.promptList dict (chat_no -> model -> messages)
  is now handled by LangGraph's checkpointer, keyed by thread_id.
- The per-call `tool_arguments` plumbing from tool_calling.py and
  chatbot_multimodal.py's stream_with_tools() is now just fields on this state.

destination_city was removed when the ticket-price tool was removed -
image_prompt (set by the generate_image tool) is the only thing that
drives image generation.

file_content has since been removed too, now that RAG is in place.
Uploaded files are chunked/embedded at upload time straight into a
per-chat Chroma collection (rag.py), and retrieve_context (tools.py) pulls
back relevant chunks per query - there's no longer a "pending file text"
value that needs to ride along in graph state waiting to be folded into
the next message.

video_id (video.py's play_video tool) is intentionally NOT reset every
turn in prepare_input, unlike image_prompt/image_b64/audio_bytes. Those
are one-shot outputs for a single turn; a loaded video should keep
playing across the rest of the conversation until a new one replaces it.
It still can't leak between chats, because state is checkpointed per
thread_id - each chat has its own persisted video_id, the same way it has
its own message history.
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    # add_messages appends new messages and de-dupes by id, same role LangGraph
    # gives every chat graph - this is what replaces mSeries.promptList[chat_no][model]
    messages: Annotated[list, add_messages]

    # Controls, one per turn (mirrors the Gradio gr.State values in chatbot.py /
    # chatbot_multimodal.py)
    source: str          # 'openRouter' | 'gemini' | 'ollama' | 'openai'
    model: str
    temperature: float

    # Feature toggles (the checkboxes in chatbot_multimodal.py)
    use_tools: bool
    speak_enabled: bool

    # The actual prompt generate_image (the graph node) will use. Set by the
    # tools node from the generate_image tool call's `prompt` argument.
    # Reset to None at the start of every turn in prepare_input so a stale
    # prompt from a previous turn never re-triggers image generation.
    image_prompt: Optional[str]

    # Outputs consumed by the Gradio UI after a run
    image_b64: Optional[str]
    audio_bytes: Optional[bytes]

    # Set by the tools node when play_video fires. Persists across turns
    # (not cleared in prepare_input) until a new video is loaded - see the
    # module docstring above for why this one field behaves differently
    # from image_b64/audio_bytes.
    video_id: Optional[str]
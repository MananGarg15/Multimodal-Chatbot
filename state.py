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

image_b64 and video_id both persist across turns rather than resetting
every turn - a generated image or a loaded video stays visible through
the rest of the conversation until a new one replaces it (image_b64 when
generate_image fires again, video_id when play_video fires again and
passes the embeddability check in video.py). image_prompt and audio_bytes
stay one-shot: image_prompt is just this turn's trigger for whether to
generate at all, and audio_bytes is a TTS clip meant to play once per
turn it's produced. See graph.py's prepare_input/call_tools for where
this split is enforced.
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

    # image_b64: persists across turns (NOT reset in prepare_input) - a
    # generated image stays visible until generate_image produces a new
    # one, same reasoning as video_id below. audio_bytes stays one-shot -
    # reset every turn, since a TTS clip is meant to play once per turn
    # it's produced, not linger as a "now showing" panel the way an image
    # or video does.
    image_b64: Optional[str]
    audio_bytes: Optional[bytes]

    # Set by the tools node when play_video fires and the video is
    # confirmed embeddable (video.py's is_embeddable). Persists across
    # turns like image_b64 - NOT reset in prepare_input - so a loaded
    # video keeps playing through the rest of the conversation until a
    # new one replaces it. Can't leak between chats since state is
    # checkpointed per thread_id; app.py reads each chat's own value back
    # on chat switch/reload the same way it already does for messages.
    video_id: Optional[str]
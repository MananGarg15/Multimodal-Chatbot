"""
Shared graph state for the combined multimodal / multi-chat bot.

Replaces:
- Day4LLMCalling/messageSeries.py's mSeries.promptList dict (chat_no -> model -> messages)
  is now handled by LangGraph's checkpointer, keyed by thread_id.
- The per-call `tool_arguments` / `destination_city` plumbing from tool_calling.py
  and chatbot_multimodal.py's stream_with_tools() is now just fields on this state.
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

    # Set by prepare_input from an uploaded PDF/TXT, consumed then cleared
    file_content: str

    # Feature toggles (the checkboxes in chatbot_multimodal.py)
    use_tools: bool
    speak_enabled: bool

    # Set by the tools node whenever set_ticket_price/get_ticket_price fires with a
    # destination_city argument this turn. Reset to None at the start of every
    # turn in prepare_input so a stale city from a previous turn never
    # re-triggers image generation.
    destination_city: Optional[str]

    # The actual prompt generate_image (the graph node) will use. Set by the
    # tools node from either the new general-purpose generate_image tool call
    # (the user's own description) or from destination_city (wrapped into a
    # city-themed prompt) - whichever tool fired. Reset to None every turn in
    # prepare_input, same reason as destination_city above.
    image_prompt: Optional[str]

    # Outputs consumed by the Gradio UI after a run
    image_b64: Optional[str]
    audio_bytes: Optional[bytes]
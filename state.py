"""
Shared graph state for the combined multimodal / multi-chat bot.

Replaces:
- Day4LLMCalling/messageSeries.py's mSeries.promptList dict (chat_no -> model -> messages)
  is now handled by LangGraph's checkpointer, keyed by thread_id.
- The per-call `tool_arguments` plumbing from tool_calling.py and
  chatbot_multimodal.py's stream_with_tools() is now just fields on this state.

destination_city has been removed - it only existed to support the
ticket-price tool's image-generation side-channel, and that tool no longer
exists. image_prompt (set directly by the generate_image tool) is now the
only thing that drives image generation.
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

    # The actual prompt generate_image (the graph node) will use. Set by the
    # tools node from the generate_image tool call's `prompt` argument.
    # Reset to None at the start of every turn in prepare_input so a stale
    # prompt from a previous turn never re-triggers image generation.
    image_prompt: Optional[str]

    # Outputs consumed by the Gradio UI after a run
    image_b64: Optional[str]
    audio_bytes: Optional[bytes]
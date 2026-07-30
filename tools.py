"""
Replaces Day4LLMCalling/tool_list.py + tool_calling.py.

Ticket-price tools (get_ticket_price, set_ticket_price, and the SQLite
Tables.db backing them) have been removed entirely per your instruction -
only image generation and, now, RAG retrieval remain as tools. langgraph's
call_tools node in graph.py binds and dispatches these directly, instead of
the old hand-written price_function JSON schema + function_map +
handle_tool_call() dispatcher.
"""

from langchain_core.tools import tool


@tool
def generate_image(prompt: str) -> str:
    """Generate an image from a text description. Call this any time the
    user asks you to draw, create, generate, show, or visualize an image.
    Pass the user's description straight through as the prompt."""
    return f"Image generation requested for: {prompt}"


@tool
def retrieve_context(query: str) -> str:
    """Search the documents the user has uploaded in this chat for
    passages relevant to `query`. Call this any time the user asks a
    question that might be answered by a file they uploaded earlier in
    this conversation (e.g. "what does the document say about X",
    "summarize the report", "according to the PDF..."). Do not call this
    if no file has been mentioned or uploaded in this chat."""
    # Actual retrieval is dispatched in graph.py's call_tools, which
    # injects this chat's thread_id (not something the model itself
    # knows or should need to pass) before querying the per-chat Chroma
    # collection in rag.py. This body exists only so llm.bind_tools() has
    # a real function to introspect for the schema/docstring - graph.py
    # intercepts this tool by name and never actually calls this body.
    raise NotImplementedError("dispatched via graph.py's call_tools with thread_id injected")


all_tools = [generate_image, retrieve_context]
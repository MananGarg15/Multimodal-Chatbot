"""
Replaces Day4LLMCalling/tool_list.py + tool_calling.py.

Ticket-price tools (get_ticket_price, set_ticket_price, and the SQLite
Tables.db backing them) have been removed entirely per your instruction -
only image generation remains as a tool. langgraph's ToolNode (or our
custom call_tools node in graph.py) binds and dispatches this directly,
instead of the old hand-written price_function JSON schema + function_map +
handle_tool_call() dispatcher.
"""

from langchain_core.tools import tool


@tool
def generate_image(prompt: str) -> str:
    """Generate an image from a text description. Call this any time the
    user asks you to draw, create, generate, show, or visualize an image.
    Pass the user's description straight through as the prompt."""
    return f"Image generation requested for: {prompt}"


all_tools = [generate_image]
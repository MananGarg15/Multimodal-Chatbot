"""
Replaces Day4LLMCalling/tool_list.py + tool_calling.py.

Ticket-price tools (get_ticket_price, set_ticket_price, and the SQLite
Tables.db backing them) have been removed entirely per your instruction.
Tools are now: generate_image, retrieve_context (RAG over uploaded docs),
and web_search / fetch_page (live web search + full-page fetch, via
Tavily). langgraph's call_tools node in graph.py binds and dispatches
these directly, instead of the old hand-written price_function JSON
schema + function_map + handle_tool_call() dispatcher.
"""

import os

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient

_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Hard cap so one page (or one search response) can't blow past the
# model's context window - same concern flagged for retrieve_context, but
# more likely to bite here since a fetched page has no chunking/relevance
# scoring at all, unlike RAG's top-k similarity search.
_MAX_CHARS = 8000


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


@tool
def web_search(query: str) -> str:
    """Search the live web for current information - news, recent events,
    facts that may have changed since training, or anything not covered by
    the user's uploaded documents. Returns a handful of results (title,
    URL, short snippet). Follow up with fetch_page on a specific URL if you
    need the full content of one result rather than just its snippet."""
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is not configured - TAVILY_API_KEY is not set."

    try:
        response = _tavily_client.search(query=query, max_results=5)
    except Exception as e:  # Tavily's client can raise several exception
        # types depending on failure mode (auth, rate limit, network) -
        # surface all of them as a tool result instead of crashing the
        # agent <-> tools loop.
        return f"Web search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No web results found."

    formatted = "\n\n---\n\n".join(
        f"{r.get('title', 'Untitled')} ({r.get('url', '')})\n{r.get('content', '')}"
        for r in results
    )
    return formatted[:_MAX_CHARS]


@tool
def fetch_page(url: str) -> str:
    """Fetch a specific web page by URL and return its main readable text.
    Use this after web_search when a result's snippet isn't enough and you
    need the full content of that page."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return text[:_MAX_CHARS]


all_tools = [generate_image, retrieve_context, web_search, fetch_page]
"""
The graph that replaces the entire old architecture:

- Day4LLMCalling/callLlms.py (both callModel and callModelGenerator)
- Day4LLMCalling/messageSeries.py (mSeries.promptList) -> now the checkpointer
- Day4LLMCalling/tool_calling.py's manual tool-call while-loop -> now the
  agent <-> tools conditional-edge loop below
- chatbot_multimodal.py's hand-written stream_with_tools() -> now just the
  `agent` node, since ChatOpenAI.stream() + .bind_tools() do both at once

Ticket-price tools (and the destination_city state field they drove) have
been removed entirely. Tools are now generate_image (sets image_prompt)
and retrieve_context (RAG lookup against this chat's uploaded documents,
see rag.py).

Graph shape (unchanged since the RAG tool was added - retrieve_context
slots into the existing agent <-> tools loop, no new nodes/edges needed):

    prepare_input -> agent <-> tools
                        |
                        +-- (image_prompt set?) --> generate_image --------+
                        |                                                  |
                        +-- (no image) ------------------------------------+
                                                                          |
                                                     (speak_enabled?) --> generate_speech --> END
                                                     (else) -----------------------------------> END
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

import rag
from llm_setup import get_llm
from multimodal import generate_image_b64, generate_speech
from state import ChatState
from tools import all_tools

TOOL_MAP = {t.name: t for t in all_tools}


def prepare_input(state: ChatState) -> dict:
    """Resets image_prompt/image/audio so a stale value from a previous
    turn can't leak into this one. File ingestion now happens at upload
    time (app.py's get_file_content -> rag.ingest_text()), not here - the
    old approach of folding the entire extracted file text into the next
    HumanMessage is gone. retrieve_context (tools.py) pulls back just the
    relevant chunks per query instead, so there's no file_content field
    left to consume/clear at this point."""
    return {"image_prompt": None, "image_b64": None, "audio_bytes": None}


def agent(state: ChatState) -> dict:
    """The single node that used to need to be two functions (callModel vs.
    callModelGenerator). Streams tokens via .stream() while still ending up
    with tool_calls populated on the accumulated AIMessage, so the
    conditional edge below can route to `tools` exactly like the old
    `while response.choices[0].finish_reason == 'tool_calls'` loop did."""
    llm = get_llm(state["source"], state["model"], state.get("temperature", 1.0))
    if state.get("use_tools"):
        llm = llm.bind_tools(all_tools)

    ai_message = None
    for chunk in llm.stream(state["messages"]):
        ai_message = chunk if ai_message is None else ai_message + chunk

    return {"messages": [ai_message]}


def route_after_agent(state: ChatState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    if state.get("image_prompt"):
        return "generate_image"
    if state.get("speak_enabled"):
        return "generate_speech"
    return END


def call_tools(state: ChatState, config: RunnableConfig) -> dict:
    """Custom tool node (rather than langgraph.prebuilt.ToolNode) so we can
    also lift image_prompt out into state for the conditional edge below,
    and so retrieve_context can be dispatched with this chat's thread_id
    injected - the model itself never sees or passes thread_id, it only
    supplies `query`. config is LangGraph's standard second node argument;
    config["configurable"]["thread_id"] is the same thread_id app.py passed
    in when it called compiled_graph.stream(...)."""
    thread_id = config["configurable"]["thread_id"]
    last = state["messages"][-1]
    tool_messages = []
    image_prompt = state.get("image_prompt")

    for call in last.tool_calls:
        if call["name"] == "retrieve_context":
            result = rag.retrieve(thread_id, call["args"].get("query", ""))
        else:
            tool_fn = TOOL_MAP[call["name"]]
            result = tool_fn.invoke(call["args"])
            if call["name"] == "generate_image":
                image_prompt = call["args"].get("prompt")

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"messages": tool_messages, "image_prompt": image_prompt}


def generate_image(state: ChatState) -> dict:
    b64 = generate_image_b64(state.get("image_prompt"))
    return {"image_b64": b64}


def route_after_image(state: ChatState) -> str:
    return "generate_speech" if state.get("speak_enabled") else END


def generate_speech_node(state: ChatState) -> dict:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content), None
    )
    text = last_ai.content if last_ai else ""
    audio = generate_speech(text)
    return {"audio_bytes": audio}


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("prepare_input", prepare_input)
    graph.add_node("agent", agent)
    graph.add_node("tools", call_tools)
    graph.add_node("generate_image", generate_image)
    graph.add_node("generate_speech", generate_speech_node)

    graph.set_entry_point("prepare_input")
    graph.add_edge("prepare_input", "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "generate_image": "generate_image",
                                      "generate_speech": "generate_speech", END: END}
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "generate_image", route_after_image, {"generate_speech": "generate_speech", END: END}
    )
    graph.add_edge("generate_speech", END)

    # SqliteSaver instead of MemorySaver: chat history now survives a
    # restart, persisted to a local file instead of living only in process
    # memory. check_same_thread=False because Gradio serves requests from a
    # worker-thread pool, not the main thread that opens this connection.
    conn = sqlite3.connect("chatbot_checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()  # creates the checkpoint tables on first run; no-op after
    return graph.compile(checkpointer=checkpointer)


compiled_graph = build_graph()
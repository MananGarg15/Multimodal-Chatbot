"""
The graph that replaces the entire old architecture:

- Day4LLMCalling/callLlms.py (both callModel and callModelGenerator)
- Day4LLMCalling/messageSeries.py (mSeries.promptList) -> now the checkpointer
- Day4LLMCalling/tool_calling.py's manual tool-call while-loop -> now the
  agent <-> tools conditional-edge loop below
- chatbot_multimodal.py's hand-written stream_with_tools() -> now just the
  `agent` node, since ChatOpenAI.stream() + .bind_tools() do both at once

Graph shape (as agreed):

    prepare_input -> agent <-> tools
                        |
                        +-- (image_prompt set?) --> generate_image --------+
                        |                                                  |
                        +-- (no image) ------------------------------------+
                                                                          |
                                                     (speak_enabled?) --> generate_speech --> END
                                                     (else) -----------------------------------> END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm_setup import get_llm
from multimodal import generate_image_b64, generate_speech
from state import ChatState
from tools import all_tools

TOOL_MAP = {t.name: t for t in all_tools}


def prepare_input(state: ChatState) -> dict:
    """Folds uploaded file content into the latest human message, same as
    `message += f'File content: {file_content}'` in chatbot.py/chatbot_multimodal.py.
    Also resets destination_city/image/audio so a stale value from a previous
    turn can't leak into this one."""
    updates: dict = {"destination_city": None, "image_prompt": None, "image_b64": None, "audio_bytes": None}

    file_content = state.get("file_content", "")
    if file_content:
        messages = state["messages"]
        if messages and isinstance(messages[-1], HumanMessage):
            last = messages[-1]
            appended = HumanMessage(
                content=f"{last.content}\n\nFile content: {file_content}", id=last.id
            )
            updates["messages"] = [appended]  # same id -> add_messages replaces in place
        updates["file_content"] = ""

    return updates


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


def call_tools(state: ChatState) -> dict:
    """Custom tool node (rather than langgraph.prebuilt.ToolNode) so we can
    also lift image_prompt out into state - same trick tool_calling.py's
    handle_tool_call() + chatbot_multimodal.py's
    tool_argument.get('destination_city') did, now generalized so either the
    ticket-price tool (destination_city -> a city-themed prompt) or the new
    general-purpose generate_image tool (the user's own prompt, untouched)
    can trigger image generation."""
    last = state["messages"][-1]
    tool_messages = []
    destination_city = state.get("destination_city")
    image_prompt = state.get("image_prompt")

    for call in last.tool_calls:
        tool_fn = TOOL_MAP[call["name"]]
        result = tool_fn.invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        if call["name"] == "generate_image":
            image_prompt = call["args"].get("prompt")
        elif call["args"].get("destination_city"):
            destination_city = call["args"]["destination_city"]
            image_prompt = f"Generate an artistic image for {destination_city}"

    return {"messages": tool_messages, "destination_city": destination_city, "image_prompt": image_prompt}


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

    # MemorySaver now, swap for langgraph.checkpoint.sqlite.SqliteSaver later
    # so chats survive a restart - same upgrade path we discussed for
    # replacing mSeries.promptList.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


compiled_graph = build_graph()
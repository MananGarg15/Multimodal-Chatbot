"""
The graph that replaces the entire old architecture:

- Day4LLMCalling/callLlms.py (both callModel and callModelGenerator)
- Day4LLMCalling/messageSeries.py (mSeries.promptList) -> now the checkpointer
- Day4LLMCalling/tool_calling.py's manual tool-call while-loop -> now the
  agent <-> tools conditional-edge loop below
- chatbot_multimodal.py's hand-written stream_with_tools() -> now just the
  `agent` node, since ChatOpenAI.stream() + .bind_tools() do both at once

Ticket-price tools (and the destination_city state field they drove) have
been removed entirely. Tools are now generate_image (sets image_prompt),
retrieve_context (RAG lookup against this chat's uploaded documents, see
rag.py), web_search/fetch_page (Tavily), and search_youtube/
extract_youtube_transcript/play_video (video.py - play_video sets
video_id, checked against YouTube's oEmbed endpoint first since some
videos have embedding disabled by their uploader).

image_b64 and video_id both persist across turns now, rather than being
reset every turn the way image_prompt/audio_bytes still are. A generated
image or a loaded video stays visible through the rest of the
conversation until a new one replaces it - each is only overwritten when
its producing node/tool actually fires again (generate_image, play_video),
never implicitly cleared just because another turn happened. app.py reads
both back per-thread on chat switch/reload the same way it already does
for message history, so they can't leak between chats either - each
chat's checkpoint carries its own image_b64/video_id.

Graph shape (unchanged since the RAG/web-search/video tools were added -
they all slot into the existing agent <-> tools loop, no new nodes/edges
needed):

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
from video import extract_video_id, fetch_transcript, is_embeddable

TOOL_MAP = {t.name: t for t in all_tools}


def prepare_input(state: ChatState) -> dict:
    """Resets image_prompt/audio_bytes so a stale value from a previous
    turn can't leak into this one - image_prompt is just this turn's
    trigger (re-checked every turn via route_after_agent), and audio_bytes
    is a one-shot TTS output meant to play once per turn it's produced.

    image_b64 and video_id are deliberately NOT reset here - both persist
    across turns until their producing node/tool fires again (see the
    module docstring above). Clearing them on every turn was tried and
    caused more confusion than it solved (a fresh turn wiping out a video/
    image the user was still looking at), so this only resets the two
    genuinely one-shot fields.

    File ingestion happens at upload time (app.py's get_file_content ->
    rag.ingest_text()), not here - the old approach of folding the entire
    extracted file text into the next HumanMessage is gone.
    retrieve_context (tools.py) pulls back just the relevant chunks per
    query instead, so there's no file_content field left to consume/clear
    at this point."""
    return {"image_prompt": None, "audio_bytes": None}


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
    also lift image_prompt/video_id out into state for the conditional
    edge below and the UI respectively, and so retrieve_context can be
    dispatched with this chat's thread_id injected - the model itself
    never sees or passes thread_id, it only supplies `query`. config is
    LangGraph's standard second node argument;
    config["configurable"]["thread_id"] is the same thread_id app.py passed
    in when it called compiled_graph.stream(...).

    video_id starts from state.get("video_id") - the value persisted from
    whenever play_video last set it (possibly several turns ago, possibly
    never), since prepare_input no longer resets it. It only changes here
    if play_video fires again this turn and passes the is_embeddable
    check below; otherwise the existing value is returned unchanged, which
    is what keeps a loaded video visible across turns that don't touch it.

    play_video is special-cased before tool_fn.invoke() even runs (unlike
    generate_image, which still calls its own tool body) because deciding
    what actually happened requires a network call (is_embeddable) the
    tool function itself has no business making - some videos have
    embedding disabled by their uploader and will never play in our
    iframe no matter what, so video_id is only overwritten - and the model
    is only told "Now playing" - when the check confirms it'll actually
    work.

    extract_youtube_transcript is special-cased for the same underlying
    reason as retrieve_context: it needs thread_id injected to know which
    chat's RAG store to write into. video.fetch_transcript() is called
    directly here (rather than via tool_fn.invoke()) so the *untruncated*
    transcript is available for rag.ingest_text() - the tool's own body
    only ever returns a char_limit-truncated slice, which is enough for
    that turn's model context but would make for a useless, already-
    truncated RAG index if that's all we ingested."""
    thread_id = config["configurable"]["thread_id"]
    last = state["messages"][-1]
    tool_messages = []
    image_prompt = state.get("image_prompt")
    video_id = state.get("video_id")

    for call in last.tool_calls:
        if call["name"] == "retrieve_context":
            result = rag.retrieve(thread_id, call["args"].get("query", ""))
        elif call["name"] == "extract_youtube_transcript":
            # Special-cased for the same reason play_video is: this needs
            # thread_id (to know which chat's RAG store to write into) and
            # the *untruncated* transcript text, neither of which the
            # tool's own body (video.py) has access to or should need to.
            resolved = extract_video_id(call["args"].get("video_id", "")) or call["args"].get(
                "video_id", ""
            )
            char_limit = call["args"].get("char_limit", 3000)
            full_text, error = fetch_transcript(resolved)
            if error:
                result = error
            else:
                chunk_count = rag.ingest_text(thread_id, f"youtube_{resolved}", full_text)
                result = full_text[:char_limit]
                if chunk_count:
                    result += (
                        f"\n\n[Full transcript indexed as {chunk_count} chunks in this chat's "
                        f"document store - use retrieve_context for anything beyond what's shown above.]"
                    )
        elif call["name"] == "play_video":
            # Don't trust play_video's own return value for this one - it
            # can't know whether the video is actually embeddable, that
            # check needs a network call, done here rather than inside the
            # tool body so the tool stays a plain, testable function.
            resolved = extract_video_id(call["args"].get("video_id", ""))
            if not resolved:
                result = f"Could not extract a valid YouTube video id from: {call['args'].get('video_id', '')}"
            elif is_embeddable(resolved):
                video_id = resolved
                result = f"Now playing video {resolved}."
            else:
                result = (
                    f"Video {resolved} cannot be played here - the uploader has "
                    f"disabled embedding for this video. Tell the user to watch "
                    f"it directly at https://www.youtube.com/watch?v={resolved}"
                )
        else:
            tool_fn = TOOL_MAP[call["name"]]
            result = tool_fn.invoke(call["args"])
            if call["name"] == "generate_image":
                image_prompt = call["args"].get("prompt")

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"messages": tool_messages, "image_prompt": image_prompt, "video_id": video_id}


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
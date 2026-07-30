"""
Replaces Week2LLMFrameworks/chatbot.py + chatbot_multimodal.py's Gradio UI.

Key change from the old UI: each sidebar "chat" is now a thread_id passed to
compiled_graph via config={"configurable": {"thread_id": ...}}. The
checkpointer (SqliteSaver in graph.py) reloads that thread's full message
history automatically - there is no more mSeries.promptList[chat_no][model]
dict to manage by hand, and per your call, history is shared across models
within a chat rather than split per-model.

Sidebar chats are now {"id": ..., "name": ...} dicts with stable,
never-reused ids (see chat_store.py), persisted to chat_list.json so the
sidebar remembers every chat across a restart - previously it only ever
started back at ["Chat1"] even though the sqlite checkpoint file still had
every thread's history sitting in it.

The generated-image and video panels both persist across turns now (see
state.py/graph.py) - an image or video stays visible until a new one
replaces it, rather than clearing on the next message. Switching chats,
deleting the current chat, or refreshing the browser all read each
thread's own image_b64/video_id back from the checkpointer (_thread_image,
_thread_video_html below) the same way message history already gets
restored, so nothing leaks between chats.
"""

import base64
import copy
import io
from pathlib import Path

import gradio as gr
import PyPDF2
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage

import chat_store
import rag
from graph import compiled_graph
from llm_setup import DEFAULT_MODEL_ALIASES


def get_file_content(file_path, chat_id):
    """Extracts text from an uploaded PDF/TXT and embeds it straight into
    this chat's Chroma collection via rag.ingest_text() - replacing the old
    approach of stashing the raw text in a file_content state value that
    got dumped wholesale into the very next message. The file's content
    now stays queryable (via the retrieve_context tool) for the rest of
    the chat instead of just the next turn, and a large document no longer
    risks blowing past the model's context window in one shot."""
    filename = Path(file_path.name).name

    if file_path.name.endswith(".pdf"):
        with open(file_path.name, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.name.endswith(".txt"):
        with open(file_path.name, "r") as f:
            text = f.read()
    else:
        return "⚠️ Unsupported file type."

    chunk_count = rag.ingest_text(chat_id, filename, text)
    if chunk_count == 0:
        return f"⚠️ Couldn't extract any text from {filename}."
    return f"📄 Indexed {filename} ({chunk_count} chunks) - ask me about it."


def _history_to_gradio(thread_id, source, model, temperature, use_tools, speak_enabled):
    """Read back a thread's message list for display when switching chats,
    replacing chatbot.py's update_chatbox(). Returns Gradio 6 "messages"
    format - a flat list of {"role", "content"} dicts - not the old
    [[user, bot], ...] tuple-pairs format, which gr.Chatbot no longer accepts."""
    config = {"configurable": {"thread_id": str(thread_id)}}
    snapshot = compiled_graph.get_state(config)
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            out.append({"role": "assistant", "content": m.content})
    return out


def _video_html(video_id):
    """Builds the embedded-player iframe for a given video_id, or a
    placeholder panel (matching image_box's footprint) when nothing's
    loaded yet - an empty string here would render as literally nothing,
    which is why the video panel used to disappear entirely instead of
    sitting there empty the way image_box always does."""
    if not video_id:
        return (
            '<div style="height:320px; display:flex; align-items:center; '
            'justify-content:center; border-radius:var(--radius-lg); '
            'background:var(--block-background-fill); '
            'border:1px solid var(--border-color-primary); '
            'color:var(--body-text-color-subdued); font-size:0.9em;">'
            "No video loaded</div>"
        )
    return (
        f'<iframe width="100%" height="320" style="border-radius:var(--radius-lg);" '
        f'src="https://www.youtube.com/embed/{video_id}" '
        f'title="YouTube video player" frameborder="0" '
        f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
        f'gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'
    )


def _thread_video_html(thread_id):
    """Reads back this thread's persisted video_id (see state.py/graph.py
    for why video_id, like image_b64, survives across turns) so
    switching to a chat that already had a video loaded shows it again,
    the same way switching chats already restores message history."""
    config = {"configurable": {"thread_id": str(thread_id)}}
    snapshot = compiled_graph.get_state(config)
    video_id = snapshot.values.get("video_id") if snapshot.values else None
    return _video_html(video_id)


def _thread_image(thread_id):
    """Same idea as _thread_video_html, for image_b64. Both fields persist
    in graph state across turns now (see state.py), so both need this
    same read-back-on-switch treatment - restoring one but not the other
    would mean a generated image quietly vanishes when you leave and
    return to a chat, while a loaded video wouldn't."""
    config = {"configurable": {"thread_id": str(thread_id)}}
    snapshot = compiled_graph.get_state(config)
    image_b64 = snapshot.values.get("image_b64") if snapshot.values else None
    return Image.open(io.BytesIO(base64.b64decode(image_b64))) if image_b64 else None


def run_turn(message, source, model, temperature, chat_id, use_tools, speak_enabled, history):
    """Streams the assistant reply token-by-token via stream_mode='messages',
    then yields the finished image/audio once the graph reaches END. This
    replaces wrapLlm() in both chatbot.py and chatbot_multimodal.py, and
    stream_with_tools() in chatbot_multimodal.py. No file_content here
    anymore - uploaded files are ingested straight into this chat's Chroma
    collection at upload time (see get_file_content above), and pulled back
    per-query by the retrieve_context tool instead of riding along on every
    turn's state."""
    config = {"configurable": {"thread_id": str(chat_id)}}
    inputs = {
        "messages": [HumanMessage(content=message)],
        "source": source,
        "model": model,
        "temperature": temperature,
        "use_tools": use_tools,
        "speak_enabled": speak_enabled,
    }

    history = copy.deepcopy(history) or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for token_chunk, metadata in compiled_graph.stream(inputs, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "agent" and getattr(token_chunk, "content", None):
            history[-1]["content"] += token_chunk.content
            # gr.skip() on the image/video panels during streaming - both
            # persist across turns now (see state.py), so there's no
            # "this turn's image/video" to show partway through; only the
            # final state below reflects whatever's actually loaded right
            # now, whether this turn changed it or not.
            yield history, gr.skip(), None, gr.skip()

    final_state = compiled_graph.get_state(config).values
    image_b64 = final_state.get("image_b64")
    audio_bytes = final_state.get("audio_bytes")
    video_id = final_state.get("video_id")

    # gr.Image doesn't accept raw bytes - it wants a numpy.ndarray,
    # PIL.Image, or a file path/string. Decoding straight to PIL here
    # (rather than passing base64.b64decode(image_b64) through as-is) is
    # what fixes the "Cannot process this value as an Image, it is of
    # type: <class 'bytes'>" ComponentProcessingError.
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))) if image_b64 else None

    yield history, image, audio_bytes, _video_html(video_id)


def load_chat(chat_id, chat_name, source, model, temperature, use_tools, speak_enabled):
    history = _history_to_gradio(chat_id, source, model, temperature, use_tools, speak_enabled)
    return chat_id, gr.update(value=history, label=chat_name), _thread_image(chat_id), _thread_video_html(chat_id)


def create_new_chat(chat_list):
    chat_list = copy.deepcopy(chat_list)
    new_id = chat_store.next_chat_id(chat_list)
    new_name = f"Chat{new_id}"
    chat_list.append({"id": new_id, "name": new_name})
    chat_store.save_chats(chat_list)
    return chat_list, new_id, gr.update(value=[], label=new_name), None, _video_html(None)


def delete_chat(chat_list, chat_id, current_chat_id):
    """Removes a chat from the sidebar, clears its persisted thread from
    the sqlite checkpointer, and re-saves the sidebar list so the deletion
    survives a restart too (not just the current session)."""
    chat_list = copy.deepcopy(chat_list)
    chat_list = [c for c in chat_list if c["id"] != chat_id]

    try:
        compiled_graph.checkpointer.delete_thread(str(chat_id))
    except AttributeError:
        # Older langgraph versions don't expose delete_thread; the row
        # disappears from the sidebar either way, this just means the raw
        # rows linger in the sqlite file until manually cleaned up.
        pass
    rag.delete_thread_store(str(chat_id))

    if not chat_list:
        chat_list = [{"id": "1", "name": "Chat1"}]

    chat_store.save_chats(chat_list)

    still_exists = any(c["id"] == current_chat_id for c in chat_list)
    if chat_id == current_chat_id or not still_exists:
        new_chat = chat_list[0]
        new_history = _history_to_gradio(new_chat["id"], None, None, None, None, None)
        return (
            chat_list,
            new_chat["id"],
            gr.update(value=new_history, label=new_chat["name"]),
            _thread_image(new_chat["id"]),
            _thread_video_html(new_chat["id"]),
        )
    return chat_list, current_chat_id, gr.skip(), gr.skip(), gr.skip()


def start_rename(chat_id, chat_name):
    """Opens the (always-present, normally hidden) rename bar for a given
    chat, pre-filled with its current name."""
    return chat_id, chat_name, gr.update(visible=True)


def cancel_rename():
    return None, gr.update(visible=False)


def save_rename(chat_list, chat_id, new_name, current_chat_id):
    chat_list = copy.deepcopy(chat_list)
    new_name = (new_name or "").strip()
    label_update = gr.skip()
    if new_name:
        for c in chat_list:
            if c["id"] == chat_id:
                c["name"] = new_name
                break
        chat_store.save_chats(chat_list)
        if chat_id == current_chat_id:
            # Renamed the chat you're currently looking at - update the
            # label in place without touching the message list itself.
            label_update = gr.update(label=new_name)
    return chat_list, None, gr.update(visible=False), label_update


def on_page_load():
    """Fired by demo.load() below - runs fresh every time a browser tab
    opens or refreshes the app, unlike a gr.State's default value (which is
    baked in once, when build_app() runs at server startup, and then reused
    - stale - for every session after that). This is what makes chats
    created/renamed/deleted in one session still be there after a refresh,
    instead of the sidebar reverting to whatever existed when the python
    process first started."""
    chats = chat_store.load_chats()
    first = chats[0]
    history = _history_to_gradio(first["id"], None, None, None, None, None)
    return chats, first["id"], gr.update(value=history, label=first["name"]), _thread_image(first["id"]), _thread_video_html(first["id"])


def reset_content():
    return ""


def on_source_change(new_source):
    """Whenever the source dropdown changes, snap model_name (state) and
    model_name_input (the textbox the user sees) to that source's correct
    default model - this is the fix for the bug where a stale model string
    from the previous source was silently sent to the new source's API."""
    default_model = DEFAULT_MODEL_ALIASES.get(new_source, "")
    return new_source, default_model, default_model


def build_app():
    with gr.Blocks() as demo:
        # Empty/placeholder defaults here - the real values get filled in by
        # demo.load(on_page_load, ...) below, which runs fresh every time a
        # browser session starts. Do NOT call chat_store.load_chats() here
        # directly: build_app() only runs once, at server startup, so a
        # value baked in at this point would be reused - stale - by every
        # session/refresh after the first.
        chat_list = gr.State([])
        chat_no = gr.State(None)
        rename_target = gr.State(None)  # id of the chat currently being renamed, if any
        source = gr.State("openai")
        model_name = gr.State(DEFAULT_MODEL_ALIASES["openai"])
        temperature = gr.State(0.0)
        use_tools = gr.State(True)
        speak_enabled = gr.State(False)

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                new_chat_btn = gr.Button("New", variant="huggingface", size="sm")

                @gr.render(inputs=[chat_list, chat_no])
                def render_chats(chat_list_value, current_chat_id):
                    with gr.Group():
                        for chat in chat_list_value:
                            cid = gr.State(chat["id"])
                            cname = gr.State(chat["name"])
                            is_active = chat["id"] == current_chat_id
                            with gr.Row():
                                btn = gr.Button(
                                    chat["name"],
                                    size="lg",
                                    variant="primary" if is_active else "secondary",
                                    scale=6,
                                )
                                rename_btn = gr.Button("✏️", size="sm", scale=1, min_width=36)
                                del_btn = gr.Button("🗑", size="sm", scale=1, min_width=36)
                            btn.click(
                                fn=load_chat,
                                inputs=[cid, cname, source, model_name, temperature, use_tools, speak_enabled],
                                outputs=[chat_no, chat_history, image_box, video_box],
                            )
                            rename_btn.click(
                                fn=start_rename,
                                inputs=[cid, cname],
                                outputs=[rename_target, rename_box, rename_row],
                            )
                            del_btn.click(
                                fn=delete_chat,
                                inputs=[chat_list, cid, chat_no],
                                outputs=[chat_list, chat_no, chat_history, image_box, video_box],
                            )

                # Always present but hidden until a ✏️ is clicked - a single
                # shared rename bar instead of swapping a row's contents
                # in-place, so there's always a real textbox on the page
                # to type into rather than one appearing/disappearing as
                # part of the per-row @gr.render loop above.
                with gr.Row(visible=False) as rename_row:
                    rename_box = gr.Textbox(show_label=False, container=False, placeholder="New name", scale=4)
                    rename_save_btn = gr.Button("✔", size="sm", scale=1, min_width=36)
                    rename_cancel_btn = gr.Button("✕", size="sm", scale=1, min_width=36)

                rename_cancel_btn.click(fn=cancel_rename, outputs=[rename_target, rename_row])

            with gr.Column(scale=4):
                chat_history = gr.Chatbot(label="Chat History", height=430)

                with gr.Row():
                    image_box = gr.Image(height=320, interactive=False, show_label=False, label="Generated image")
                    audio_box = gr.Audio(autoplay=True, label="Voice reply")
                    video_box = gr.HTML(_video_html(None), label="Video", min_height=320, container=True)

                # Both wired here rather than right after their buttons'
                # own definitions above - these .click() calls run eagerly
                # at build time (unlike the @gr.render closures above,
                # which are deferred until actual rendering), so they need
                # chat_history/image_box/video_box to already exist as
                # Python names in this scope, not just be defined
                # somewhere later in the function.
                new_chat_btn.click(
                    fn=create_new_chat,
                    inputs=[chat_list],
                    outputs=[chat_list, chat_no, chat_history, image_box, video_box],
                )
                rename_save_btn.click(
                    fn=save_rename,
                    inputs=[chat_list, rename_target, rename_box, chat_no],
                    outputs=[chat_list, rename_target, rename_row, chat_history],
                )

                with gr.Group():
                    user_input = gr.Textbox(placeholder="Enter your prompt", show_label=False, scale=8)
                    submit_btn = gr.Button("enter", size="sm", variant="primary", scale=1)

                turn_inputs = [
                    user_input, source, model_name, temperature,
                    chat_no, use_tools, speak_enabled, chat_history,
                ]
                turn_outputs = [chat_history, image_box, audio_box, video_box]

                submit_btn.click(run_turn, inputs=turn_inputs, outputs=turn_outputs).then(
                    fn=reset_content, outputs=[user_input]
                )
                user_input.submit(run_turn, inputs=turn_inputs, outputs=turn_outputs).then(
                    fn=reset_content, outputs=[user_input]
                )

            with gr.Column(scale=1):
                with gr.Accordion("Adv_settings"):
                    source_selection = gr.Dropdown(
                        choices=["openRouter", "gemini", "ollama", "openai"], value="openai", label="Select source"
                    )

                    temperature_select = gr.Slider(0, 2, value=0, step=0.1, label="temp_slider")
                    temperature_select.change(fn=lambda t: t, inputs=[temperature_select], outputs=[temperature])

                    model_name_input = gr.Textbox(
                        placeholder="Enter model name", value=DEFAULT_MODEL_ALIASES["openai"], label="Enter Model name"
                    )
                    model_name_input.submit(fn=lambda m: m, inputs=[model_name_input], outputs=[model_name])

                    # Declared after model_name_input so it can appear in this
                    # handler's outputs; must run after the textbox exists.
                    source_selection.change(
                        fn=on_source_change,
                        inputs=[source_selection],
                        outputs=[source, model_name, model_name_input],
                    )

                    tools_checkbox = gr.Checkbox(
                        label="Enable tools (image, video, web search, document search)", value=True
                    )
                    tools_checkbox.change(fn=lambda t: t, inputs=[tools_checkbox], outputs=[use_tools])

                    speak_checkbox = gr.Checkbox(label="Speak replies aloud (TTS)", value=False)
                    speak_checkbox.change(fn=lambda t: t, inputs=[speak_checkbox], outputs=[speak_enabled])

                files = gr.File(label="insert file", file_count="single", file_types=[".pdf", ".txt"])
                upload_status = gr.Markdown()
                files.upload(fn=get_file_content, inputs=[files, chat_no], outputs=[upload_status])

        # Fires once per browser session (initial load AND every refresh) -
        # this is what actually re-reads chat_list.json each time, unlike
        # the gr.State defaults above which are fixed once at server
        # startup. This is the fix for chats disappearing on refresh.
        demo.load(fn=on_page_load, outputs=[chat_list, chat_no, chat_history, image_box, video_box])

    return demo


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Soft())
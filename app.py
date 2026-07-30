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
"""

import base64
import copy
import io

import gradio as gr
import PyPDF2
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage

import chat_store
from graph import compiled_graph
from llm_setup import DEFAULT_MODEL_ALIASES


def get_file_content(file_path):
    if file_path.name.endswith(".pdf"):
        with open(file_path.name, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() or "" for page in reader.pages)
    if file_path.name.endswith(".txt"):
        with open(file_path.name, "r") as f:
            return f.read()
    return ""


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


def run_turn(message, file_content, source, model, temperature, chat_id, use_tools, speak_enabled, history):
    """Streams the assistant reply token-by-token via stream_mode='messages',
    then yields the finished image/audio once the graph reaches END. This
    replaces wrapLlm() in both chatbot.py and chatbot_multimodal.py, and
    stream_with_tools() in chatbot_multimodal.py."""
    config = {"configurable": {"thread_id": str(chat_id)}}
    inputs = {
        "messages": [HumanMessage(content=message)],
        "source": source,
        "model": model,
        "temperature": temperature,
        "file_content": file_content,
        "use_tools": use_tools,
        "speak_enabled": speak_enabled,
    }

    history = copy.deepcopy(history) or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for token_chunk, metadata in compiled_graph.stream(inputs, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "agent" and getattr(token_chunk, "content", None):
            history[-1]["content"] += token_chunk.content
            yield history, None, None

    final_state = compiled_graph.get_state(config).values
    image_b64 = final_state.get("image_b64")
    audio_bytes = final_state.get("audio_bytes")

    # gr.Image doesn't accept raw bytes - it wants a numpy.ndarray,
    # PIL.Image, or a file path/string. Decoding straight to PIL here
    # (rather than passing base64.b64decode(image_b64) through as-is) is
    # what fixes the "Cannot process this value as an Image, it is of
    # type: <class 'bytes'>" ComponentProcessingError.
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))) if image_b64 else None

    yield history, image, audio_bytes


def load_chat(chat_id, source, model, temperature, use_tools, speak_enabled):
    return chat_id, _history_to_gradio(chat_id, source, model, temperature, use_tools, speak_enabled)


def create_new_chat(chat_list):
    chat_list = copy.deepcopy(chat_list)
    new_id = chat_store.next_chat_id(chat_list)
    chat_list.append({"id": new_id, "name": f"Chat{new_id}"})
    chat_store.save_chats(chat_list)
    return chat_list, new_id


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

    if not chat_list:
        chat_list = [{"id": "1", "name": "Chat1"}]

    chat_store.save_chats(chat_list)

    still_exists = any(c["id"] == current_chat_id for c in chat_list)
    if chat_id == current_chat_id or not still_exists:
        new_chat_id = chat_list[0]["id"]
        new_history = _history_to_gradio(new_chat_id, None, None, None, None, None)
        return chat_list, new_chat_id, new_history
    return chat_list, current_chat_id, gr.skip()


def start_rename(chat_id, chat_name):
    """Opens the (always-present, normally hidden) rename bar for a given
    chat, pre-filled with its current name."""
    return chat_id, chat_name, gr.update(visible=True)


def cancel_rename():
    return None, gr.update(visible=False)


def save_rename(chat_list, chat_id, new_name):
    chat_list = copy.deepcopy(chat_list)
    new_name = (new_name or "").strip()
    if new_name:
        for c in chat_list:
            if c["id"] == chat_id:
                c["name"] = new_name
                break
        chat_store.save_chats(chat_list)
    return chat_list, None, gr.update(visible=False)


def on_page_load():
    """Fired by demo.load() below - runs fresh every time a browser tab
    opens or refreshes the app, unlike a gr.State's default value (which is
    baked in once, when build_app() runs at server startup, and then reused
    - stale - for every session after that). This is what makes chats
    created/renamed/deleted in one session still be there after a refresh,
    instead of the sidebar reverting to whatever existed when the python
    process first started."""
    chats = chat_store.load_chats()
    first_id = chats[0]["id"]
    history = _history_to_gradio(first_id, None, None, None, None, None)
    return chats, first_id, history


def reset_content():
    return "", ""


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
        source = gr.State("openRouter")
        model_name = gr.State("openrouter/free")
        temperature = gr.State(0.0)
        file_content = gr.State("")
        use_tools = gr.State(False)
        speak_enabled = gr.State(False)

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                new_chat_btn = gr.Button("New", variant="huggingface", size="sm")

                @gr.render(inputs=[chat_list])
                def render_chats(chat_list_value):
                    with gr.Group():
                        for chat in chat_list_value:
                            cid = gr.State(chat["id"])
                            cname = gr.State(chat["name"])
                            with gr.Row():
                                btn = gr.Button(chat["name"], size="lg", variant="stop", scale=6)
                                rename_btn = gr.Button("✏️", size="sm", scale=1, min_width=36)
                                del_btn = gr.Button("🗑", size="sm", scale=1, min_width=36)
                            btn.click(
                                fn=load_chat,
                                inputs=[cid, source, model_name, temperature, use_tools, speak_enabled],
                                outputs=[chat_no, chat_history],
                            )
                            rename_btn.click(
                                fn=start_rename,
                                inputs=[cid, cname],
                                outputs=[rename_target, rename_box, rename_row],
                            )
                            del_btn.click(
                                fn=delete_chat,
                                inputs=[chat_list, cid, chat_no],
                                outputs=[chat_list, chat_no, chat_history],
                            )

                new_chat_btn.click(fn=create_new_chat, inputs=[chat_list], outputs=[chat_list, chat_no])

                # Always present but hidden until a ✏️ is clicked - a single
                # shared rename bar instead of swapping a row's contents
                # in-place, so there's always a real textbox on the page
                # to type into rather than one appearing/disappearing as
                # part of the per-row @gr.render loop above.
                with gr.Row(visible=False) as rename_row:
                    rename_box = gr.Textbox(show_label=False, container=False, placeholder="New name", scale=4)
                    rename_save_btn = gr.Button("✔", size="sm", scale=1, min_width=36)
                    rename_cancel_btn = gr.Button("✕", size="sm", scale=1, min_width=36)

                rename_save_btn.click(
                    fn=save_rename,
                    inputs=[chat_list, rename_target, rename_box],
                    outputs=[chat_list, rename_target, rename_row],
                )
                rename_cancel_btn.click(fn=cancel_rename, outputs=[rename_target, rename_row])

            with gr.Column(scale=4):
                chat_history = gr.Chatbot(label="Chat History", height=430)

                with gr.Row():
                    image_box = gr.Image(height=320, interactive=False, show_label=False, label="Generated image")
                    audio_box = gr.Audio(autoplay=True, label="Voice reply")

                with gr.Group():
                    user_input = gr.Textbox(placeholder="Enter your prompt", show_label=False, scale=8)
                    submit_btn = gr.Button("enter", size="sm", variant="primary", scale=1)

                turn_inputs = [
                    user_input, file_content, source, model_name, temperature,
                    chat_no, use_tools, speak_enabled, chat_history,
                ]
                turn_outputs = [chat_history, image_box, audio_box]

                submit_btn.click(run_turn, inputs=turn_inputs, outputs=turn_outputs).then(
                    fn=reset_content, outputs=[file_content, user_input]
                )
                user_input.submit(run_turn, inputs=turn_inputs, outputs=turn_outputs).then(
                    fn=reset_content, outputs=[file_content, user_input]
                )

            with gr.Column(scale=1):
                with gr.Accordion("Adv_settings"):
                    source_selection = gr.Dropdown(choices=["openRouter", "gemini", "ollama", "openai"], label="Select source")

                    temperature_select = gr.Slider(0, 2, value=0, step=0.1, label="temp_slider")
                    temperature_select.change(fn=lambda t: t, inputs=[temperature_select], outputs=[temperature])

                    model_name_input = gr.Textbox(placeholder="Enter model name", value="openrouter/free", label="Enter Model name")
                    model_name_input.submit(fn=lambda m: m, inputs=[model_name_input], outputs=[model_name])

                    # Declared after model_name_input so it can appear in this
                    # handler's outputs; must run after the textbox exists.
                    source_selection.change(
                        fn=on_source_change,
                        inputs=[source_selection],
                        outputs=[source, model_name, model_name_input],
                    )

                    tools_checkbox = gr.Checkbox(label="Enable tools (image generation)", value=False)
                    tools_checkbox.change(fn=lambda t: t, inputs=[tools_checkbox], outputs=[use_tools])

                    speak_checkbox = gr.Checkbox(label="Speak replies aloud (TTS)", value=False)
                    speak_checkbox.change(fn=lambda t: t, inputs=[speak_checkbox], outputs=[speak_enabled])

                files = gr.File(label="insert file", file_count="single", file_types=[".pdf", ".txt"])
                files.upload(fn=get_file_content, inputs=[files], outputs=[file_content])

        # Fires once per browser session (initial load AND every refresh) -
        # this is what actually re-reads chat_list.json each time, unlike
        # the gr.State defaults above which are fixed once at server
        # startup. This is the fix for chats disappearing on refresh.
        demo.load(fn=on_page_load, outputs=[chat_list, chat_no, chat_history])

    return demo


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Soft())
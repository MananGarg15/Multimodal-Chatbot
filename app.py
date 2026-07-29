"""
Replaces Week2LLMFrameworks/chatbot.py + chatbot_multimodal.py's Gradio UI.

Key change from the old UI: each sidebar "chat" is now a thread_id passed to
compiled_graph via config={"configurable": {"thread_id": ...}}. The
checkpointer (MemorySaver in graph.py) reloads that thread's full message
history automatically - there is no more mSeries.promptList[chat_no][model]
dict to manage by hand, and per your call, history is shared across models
within a chat rather than split per-model.
"""

import base64
import copy

import gradio as gr
import PyPDF2
from langchain_core.messages import AIMessage, HumanMessage

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


def run_turn(message, file_content, source, model, temperature, chat_no, use_tools, speak_enabled, history):
    """Streams the assistant reply token-by-token via stream_mode='messages',
    then yields the finished image/audio once the graph reaches END. This
    replaces wrapLlm() in both chatbot.py and chatbot_multimodal.py, and
    stream_with_tools() in chatbot_multimodal.py."""
    config = {"configurable": {"thread_id": str(chat_no)}}
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

    final_state = None
    for token_chunk, metadata in compiled_graph.stream(inputs, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "agent" and getattr(token_chunk, "content", None):
            history[-1]["content"] += token_chunk.content
            yield history, None, None

    final_state = compiled_graph.get_state(config).values
    image_b64 = final_state.get("image_b64")
    audio_bytes = final_state.get("audio_bytes")
    image = base64.b64decode(image_b64) if image_b64 else None
    yield history, image, audio_bytes


def load_chat(chat_num, source, model, temperature, use_tools, speak_enabled):
    return chat_num, _history_to_gradio(chat_num, source, model, temperature, use_tools, speak_enabled)


def create_new_chat(chat_list):
    chat_list.append(f"Chat{len(chat_list) + 1}")
    return chat_list, len(chat_list)


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
        chat_list = gr.State(["Chat1"])
        chat_no = gr.State(1)
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
                def render_chats(chat_list):
                    with gr.Group():
                        for i, chat in enumerate(chat_list):
                            chat_num = gr.State(i + 1)
                            btn = gr.Button(chat, size="lg", variant="stop")
                            btn.click(
                                fn=load_chat,
                                inputs=[chat_num, source, model_name, temperature, use_tools, speak_enabled],
                                outputs=[chat_no, chat_history],
                            )

                new_chat_btn.click(fn=create_new_chat, inputs=[chat_list], outputs=[chat_list, chat_no])

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

    return demo


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Soft())
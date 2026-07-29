# LangGraph combined chatbot

Ports and merges:
- `Week2LLMFrameworks/chatbot.py` (multi-chat sidebar, streaming, file upload)
- `chatbot_multimodal.py` (tool-calling -> image gen, TTS)
- `Day4LLMCalling/*` (the old `Llms` client wrapper, `mSeries`, tool schema/dispatch)

into a single LangGraph app. See `graph.py` for the node/edge design.

## Setup

```bash
pip install -r requirements.txt
```

Same `.env` keys as before: `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`.
(`ollama` needs no key, just a running local server on `localhost:11434`.)

```bash
python app.py
```

## What changed vs. the old code, and why

| Old | New | Why |
|---|---|---|
| `mSeries.promptList[chat_no][model]` global dict | LangGraph `MemorySaver` checkpointer, keyed by `thread_id = chat_no` | Per your decision: one shared history per chat regardless of model switch. Also removes a global mutable dict as the source of truth. |
| `callModel` (tools, no stream) vs `callModelGenerator` (stream, no tools) | one `agent` node using `ChatOpenAI.bind_tools().stream()` | `.stream()` and `.bind_tools()` compose on the same object in langchain, so there's no more either/or flag. |
| `tool_calling.py` manual JSON schema + `function_map` + `handle_tool_call` while-loop | `tools.py` `@tool`-decorated functions + `agent <-> tools` conditional-edge loop in `graph.py` | Same dispatch logic, now expressed as graph control flow instead of a while-loop inside the LLM-calling function. |
| `set_ticket_price`'s hardcoded Windows DB path | relative path next to `tools.py` | Portability bug fix. |
| notebook's `talker()` reading `response.executable_ad_data` | `multimodal.py`'s `generate_speech()` reads `response.content` | That attribute doesn't exist; this was a bug in the original notebook (already independently fixed in `chatbot_multimodal.py`). |
| `callModelGenerator`'s `openai` branch calling `Llms.openRouter` | `llm_setup.py` correctly maps each source to its own client/base_url | Copy-paste bug fix. |
| TTS built (`talker()`) but never wired into the Gradio pipeline | `generate_speech` is a real graph node, reachable via the `speak_enabled` toggle | Feature was present but dead code before. |

## Known limitations to verify once you have keys/network

- Not run end-to-end in this sandbox (no network access, no API keys available here) - the code is written to match your existing patterns and compiles cleanly, but please smoke-test locally before relying on it.
- Tool-calling support varies by model/source - `ollama`'s default `gpt-oss:20b` and some OpenRouter free models may not reliably emit `tool_calls`; this mirrors a limitation that existed in the old code too.
- Image generation and TTS always use the native `openai` client (same as the old `artist()`/`talker()`), regardless of which `source` you're chatting with - so those features need `OPENAI_API_KEY` set even if you're chatting via Gemini/OpenRouter/Ollama.

## Next steps (per your original plan)

- Swap `MemorySaver` for `langgraph.checkpoint.sqlite.SqliteSaver` so chat history survives a restart.
- Add RAG: a `retrieve` node before `agent`, or a `retriever_tool` alongside `ticket_tools`.
- Add web search: wrap `Scraper.py`'s `scrape_text`/`get_all_links` as another `@tool`, same pattern as `tools.py`.

# Multimodal AI Chatbot

A multi-provider, tool-using chatbot built on [LangGraph](https://github.com/langchain-ai/langgraph). It streams responses token-by-token while still being able to call tools mid-turn - generating images, reading replies aloud, searching and watching YouTube videos, searching the live web, and answering questions about documents you've uploaded to that chat.

## Features

- **Four LLM providers** - OpenAI, Gemini, OpenRouter, and local Ollama models, switchable per turn from the UI.
- **Streaming + tool-calling in one pass** - the agent streams its answer token-by-token and can still decide mid-response to call a tool, loop back with the result, and continue.
- **Image generation** - ask the bot to draw, generate, or visualize something and it produces an image inline.
- **Text-to-speech** - toggle "speak replies aloud" to have responses read back to you.
- **YouTube integration** - search for videos, pull a video's full transcript to answer questions about its content, and load a video directly into an embedded player in the UI.
- **Live web search** - looks up current information via Tavily and can fetch and read a specific page in full.
- **Per-chat document Q&A (RAG)** - upload a PDF or TXT file and it's chunked, embedded, and stored in a per-chat vector index (Chroma), so it stays queryable for the rest of that conversation. YouTube transcripts get indexed the same way, so long transcripts remain fully searchable rather than being capped to what fits in one turn.
- **Multi-chat sidebar** - create, rename, and delete chats; the active chat is highlighted, and the chat panel's title always shows the name of whichever chat is open.
- **Persistent state** - chat history, generated images, loaded videos, and uploaded documents all survive an app restart.

## Architecture

The chatbot is a single LangGraph state machine (`graph.py`):

```
prepare_input -> agent <-> tools
                    |
                    +-- image requested? --> generate_image --+
                    |                                          |
                    +-- (no image) ---------------------------+
                                                               |
                                              speak enabled? --> generate_speech --> END
                                              (else) -----------------------------------> END
```

- **`agent`** builds the right client for the selected provider, streams the response, and (if tools are enabled) can emit tool calls.
- **`tools`** dispatches whichever tool the model called - image generation, RAG lookup, web search/page fetch, or the YouTube tools - injecting per-chat context (like `thread_id`) that the model itself never needs to know about.
- **`generate_image`** / **`generate_speech`** produce the actual image/audio output for that turn, only running when triggered.

State (`state.py`) is checkpointed per chat via a `SqliteSaver` (`chatbot_checkpoints.sqlite`), so each sidebar chat is its own independent thread with its own message history, generated image, loaded video, and uploaded documents.

### Modules

| File | Responsibility |
|---|---|
| `app.py` | Gradio UI - chat window, sidebar, settings, file upload |
| `graph.py` | The LangGraph graph: nodes, routing, checkpointing |
| `state.py` | Shared graph state schema |
| `llm_setup.py` | Per-provider chat model client factory |
| `tools.py` | `generate_image`, `retrieve_context`, `web_search`, `fetch_page` |
| `video.py` | `search_youtube`, `extract_youtube_transcript`, `play_video` |
| `multimodal.py` | Image generation and text-to-speech calls |
| `rag.py` | Per-chat document chunking, embedding, and retrieval (Chroma) |
| `chat_store.py` | Persists the sidebar's chat list (id + name) to `chat_list.json` |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with the keys you need:

```
OPENAI_API_KEY=...      # chat (openai source), image generation, TTS, embeddings - required
GOOGLE_API_KEY=...      # chat (gemini source)
OPENROUTER_API_KEY=...  # chat (openRouter source)
TAVILY_API_KEY=...      # web_search - free tier at tavily.com
YOUTUBE_API_KEY=...     # search_youtube only (a YouTube Data API v3 key)
```

`ollama` needs no key, just a running local server on `localhost:11434`. `OPENAI_API_KEY` is required regardless of which provider you're chatting with, since image generation, TTS, and document embeddings always go through OpenAI's API.

Run it:

```bash
python app.py
```

## Using it

- Pick a **source** and **model** in the settings panel, and a temperature.
- **Enable tools** to let the model call `generate_image`, `retrieve_context`, `web_search`, `fetch_page`, and the YouTube tools. Off by default means plain chat only.
- **Speak replies aloud** turns on TTS for that turn's response.
- Drop a **PDF or TXT** file in to index it for that chat - ask about it afterward and the model will pull in relevant passages automatically.
- Use the sidebar to start new chats, rename them, or delete them; each has its own independent history, image, video, and document index.

## Known limitations

- Tool-calling reliability varies by model - `ollama`'s local models and some free OpenRouter models don't always emit proper tool calls.
- Image generation, TTS, and embeddings always use OpenAI's API regardless of the selected chat provider, so `OPENAI_API_KEY` is required even when chatting via Gemini/OpenRouter/Ollama.
- Per-chat vector data lives under `./chroma_db/<thread_id>/`, alongside `chatbot_checkpoints.sqlite` and `chat_list.json` - back up or `.gitignore` all three together.
- `web_search` degrades to a "not configured" message rather than erroring if `TAVILY_API_KEY` is unset; `search_youtube` does the same for `YOUTUBE_API_KEY`. Neither blocks the rest of the app from working.
- `fetch_page`'s text extraction is a basic tag-strip via BeautifulSoup, not a full readability pass, so it can include some nav/sidebar noise on busier pages.
- `extract_youtube_transcript` depends on the video actually having captions (auto-generated or uploader-provided, in English); some videos don't and it returns a plain "unavailable" message.
- `play_video` checks whether a video actually allows embedding before claiming success - some uploaders disable it entirely, in which case the model is told to point the user to the YouTube link directly instead.
- The generated image and loaded video panels have no explicit "clear" action - each keeps showing whatever was last produced in that chat until a new one replaces it.
- The graph has no explicit cap on how many times a single turn can bounce between `agent` and `tools` beyond LangGraph's default recursion limit, which will raise rather than degrade gracefully if hit.

## Possible future work

- A token-budget-aware cap on `retrieve_context`/`web_search` output instead of the current flat character limits.
- Swap `fetch_page`'s BeautifulSoup extraction for something closer to Readability.js/`trafilatura` if page noise becomes a real problem.
- Voice input (speech-to-text), to match the existing voice output.
- An explicit iteration cap on the agent/tools loop with a graceful fallback message.
- `.env.example` listing all required keys for a fresh clone.
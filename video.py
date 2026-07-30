"""
YouTube integration: search, transcript extraction, and embedding a video
player in the UI. Mirrors the shape of multimodal.py/rag.py - a standalone
module the tool-dispatch logic in graph.py's call_tools calls into.

Three tools:
- search_youtube: Data API v3 search, returns metadata including video_id.
- extract_youtube_transcript: full transcript text for a specific video_id,
  so the model can actually read/summarize/answer questions about a video's
  content rather than just its title+description snippet. The truncated
  slice returned to the model is capped at char_limit (default 3000), but
  graph.py's call_tools also ingests the *untruncated* transcript into this
  chat's RAG store (rag.py) via fetch_transcript() below, the same way an
  uploaded PDF/TXT gets indexed - so a long video's transcript stays fully
  queryable via retrieve_context afterward, not just the first ~3000 chars.
- play_video: the one that actually surfaces something in the UI. Calling
  search_youtube alone does NOT display anything - it just gives the model
  video_ids to choose from (same relationship generate_image's tool call
  has to state.image_prompt: the tool call itself doesn't render anything,
  graph.py's call_tools lifts the relevant argument into state, and that's
  what the UI actually reads).

Why video_id needs different handling from image_prompt/image_b64/audio_bytes:
those are one-shot outputs - reset every turn in prepare_input so a stale
image from three turns ago doesn't linger. A loaded video is different: if
the user says "play the second one" and then keeps chatting, the video
should stay visible and playable through the rest of the conversation, not
disappear the moment the next message is sent. So state.video_id is
deliberately NOT reset in prepare_input - see state.py and graph.py's
prepare_input docstring. It still can't leak between chats, because
LangGraph state is checkpointed per thread_id - Chat A's video_id lives in
Chat A's checkpoint row, Chat B's in its own; app.py reads it back per
thread the same way it already does for message history.
"""

import os
import re
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool

_TRANSCRIPT_CHAR_LIMIT = 3000
_OEMBED_URL = "https://www.youtube.com/oembed"

# Matches youtu.be/<id>, youtube.com/watch?v=<id>, youtube.com/embed/<id>,
# or a bare 11-character id typed/pasted directly.
_VIDEO_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([A-Za-z0-9_-]{11})|^([A-Za-z0-9_-]{11})$"
)


def extract_video_id(url_or_id: str) -> str | None:
    """Pulls an 11-char video id out of a full YouTube URL, or passes a
    bare id straight through. Used by play_video so it works whether the
    model got the id from search_youtube's results or the user just pasted
    a link directly."""
    if not url_or_id:
        return None
    match = _VIDEO_ID_RE.search(url_or_id.strip())
    if not match:
        return None
    return match.group(1) or match.group(2)


def is_embeddable(video_id: str) -> bool:
    """Checks YouTube's oEmbed endpoint to see whether this video actually
    allows embedding, before the UI claims it's playing. Some uploaders
    disable embedding entirely (common for music videos, trailers, some
    news clips) - a disabled-embed video's iframe just shows YouTube's own
    "Video unavailable" message with a "Watch on YouTube" link, which looks
    broken but isn't a bug on our end; there's no way to override it
    client-side. Checking oEmbed first lets graph.py's call_tools give an
    honest tool result instead of silently setting video_id for a video
    that's never going to actually play in the embedded panel.

    Defaults to True (assume embeddable) on network errors/timeouts,
    rather than blocking playback attempts over a transient failure on our
    check - worst case in that scenario is the same "click through to
    YouTube" fallback a real disabled-embed video would show anyway."""
    try:
        resp = requests.get(
            _OEMBED_URL,
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
    except requests.RequestException:
        return True
    return resp.status_code == 200


@tool
def search_youtube(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Searches YouTube for videos matching a query via the Data API v3.
    Use this tool when video content (talks, demos, interviews) is likely to be
    relevant. To read a video's full transcript, pass the returned 'video_id'
    into extract_youtube_transcript. To actually show/play a video for the
    user, pass its 'video_id' into play_video.
    REQUIRES the YOUTUBE_API_KEY environment variable to be set.

    Args:
        query: The search topic or keywords.
        max_results: Maximum number of videos to retrieve (default is 10).

    Returns:
        List of dictionaries with video metadata, including 'video_id' for use
        with extract_youtube_transcript and play_video.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return [{"error": "YOUTUBE_API_KEY not set in environment."}]

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.search().list(
            q=query, part="snippet", type="video", maxResults=max_results
        ).execute()
    except Exception as e:
        return [{"error": f"Error querying YouTube API: {str(e)}"}]

    results = []
    for item in response.get("items", []):
        vid = item["id"]["videoId"]
        snippet = item["snippet"]
        results.append({
            "title": snippet.get("title", ""),
            "author": snippet.get("channelTitle"),
            "created": snippet.get("publishedAt"),
            "text": snippet.get("description"),
            "video_id": vid,
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return results


def fetch_transcript(video_id: str) -> tuple[str | None, str | None]:
    """Fetches a video's full, un-truncated transcript once. Returns
    (full_text, error) - exactly one of the two is None.

    Factored out of extract_youtube_transcript so graph.py's call_tools can
    also ingest the *full* transcript into this chat's RAG store (rag.py)
    without hitting the transcript API a second time just to get the
    untruncated text - the tool below only ever needs a truncated slice
    for the model's context, but ingestion needs the whole thing."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
    except ImportError:
        return None, "Error: youtube-transcript-api is not installed."

    try:
        ytt_api = YouTubeTranscriptApi()
        try:
            fetched = ytt_api.fetch(video_id, languages=["en"])
        except NoTranscriptFound:
            transcript_list = ytt_api.list(video_id)
            transcript = next(iter(transcript_list))
            fetched = transcript.fetch()

        text = " ".join(snippet.text for snippet in fetched)
        return text, None

    except (TranscriptsDisabled, VideoUnavailable):
        return None, "Transcript unavailable for this video (disabled by uploader or video unavailable)."
    except Exception as e:
        return None, f"Error extracting YouTube transcript: {str(e)}"


@tool
def extract_youtube_transcript(video_id: str, char_limit: int = 3000) -> str:
    """
    Fetches the transcript of a YouTube video and shows you up to
    char_limit characters of it directly in this turn's context. The full
    transcript (not just this truncated slice) also gets indexed into this
    chat's document store, so you can use retrieve_context afterward to
    search anything beyond what's shown here, rather than needing the
    whole thing to fit in context at once.
    Use this tool AFTER search_youtube, passing the 'video_id' of a specific video
    you want to read in full rather than just its description. Falls back to a
    "transcript unavailable" message if the video has no transcript (disabled by
    the uploader, or none in a usable language).
    Requires the youtube-transcript-api package to be installed.

    Args:
        video_id: The YouTube video ID, as returned by search_youtube.
        char_limit: Maximum number of characters to return (default is 3000).

    Returns:
        The video transcript as plain text, truncated to char_limit.
    """
    full_text, error = fetch_transcript(video_id)
    if error:
        return error
    return full_text[:char_limit]
    # Note: graph.py's call_tools special-cases this tool by name (same
    # pattern as play_video) so it can also call rag.ingest_text() with the
    # *full* transcript from fetch_transcript() above, and inject thread_id
    # - neither of which this body has access to or should need to. This
    # implementation is what the schema/docstring get introspected from,
    # and is a working fallback if that interception is ever bypassed, but
    # in normal operation call_tools's dispatch is what actually runs.


@tool
def play_video(video_id: str) -> str:
    """Load a YouTube video into the embedded player shown to the user.
    Call this whenever the user asks to watch, play, or see a specific
    video - either one you just found via search_youtube (pass its
    'video_id'), or one the user gave you directly as a URL or id. The
    video stays visible for the rest of the conversation until a new one
    is loaded, so you don't need to call this again on every turn.

    Note: some videos have embedding disabled by their uploader and can't
    play in the embedded panel at all - if that's the case here, the tool
    result you get back will say so instead of "Now playing", and you
    should tell the user to open it directly on YouTube instead of acting
    like it's playing.

    Args:
        video_id: A YouTube video id, or a full YouTube/youtu.be URL - both
            work, the id is extracted automatically.
    """
    resolved = extract_video_id(video_id)
    if not resolved:
        return f"Could not extract a valid YouTube video id from: {video_id}"
    # The actual state update (state.video_id) AND the embeddability check
    # (is_embeddable) happen in graph.py's call_tools, which special-cases
    # this tool by name - same pattern as generate_image/image_prompt.
    # This return value is a placeholder; call_tools overwrites the
    # ToolMessage content with the real outcome (playing vs. not
    # embeddable) once it knows which one actually happened.
    return f"Attempting to play video {resolved}."


video_tools = [search_youtube, extract_youtube_transcript, play_video]
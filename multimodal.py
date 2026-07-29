"""
Replaces artist()/talker() from notebooks/W2D5ImageAudio.ipynb.

Image model: gpt-image-2 (quality="low"), TTS: gpt-4o-mini-tts.

Notes vs. the original notebook code:
- talker() actually gets wired into the graph now (it existed in the notebook
  but was commented out of the Gradio pipeline).
- The `response.executable_ad_data` bug is fixed to `response.content`
  (chatbot_multimodal.py already made this same fix independently).
- images.generate() no longer passes response_format: the GPT image models
  (gpt-image-1-mini, gpt-image-1.5, gpt-image-2, etc.) don't accept that
  parameter at all and always return b64_json in response.data[0].b64_json.
  Passing it causes a 400 "Unknown parameter: 'response_format'" error.
- Switched from gpt-image-1-mini to gpt-image-2 with quality="low": OpenAI's
  docs note gpt-image-2 at low quality performs on par with gpt-image-1-mini,
  but gpt-image-2 is the current flagship model (not on a deprecation
  timeline the way gpt-image-1 / gpt-image-1.5 are), so this is the more
  future-proof pick at essentially the same cost.
"""

import base64

from llm_setup import openai_native_client


def generate_image_b64(prompt: str) -> str | None:
    """Returns a base64 PNG string (not a PIL.Image) so it can live directly
    in ChatState without extra serialization for the checkpointer. Takes any
    free-form prompt now - the caller (graph.py) is responsible for building
    a city-specific prompt for the ticket-tool flow, or passing the user's
    own description through for the general-purpose image tool."""
    if not prompt:
        return None
    image_response = openai_native_client.images.generate(
        prompt=prompt,
        model="gpt-image-2",
        quality="low",
        n=1,
        size="1024x1024",
    )
    return image_response.data[0].b64_json


def generate_speech(text: str) -> bytes | None:
    if not text:
        return None
    response = openai_native_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
    )
    return response.content  # was response.executable_ad_data in the notebook - not a real attribute


def b64_to_bytes(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)
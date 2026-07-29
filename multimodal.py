"""
Replaces artist()/talker() from notebooks/W2D5ImageAudio.ipynb.

Same models (gpt-image-1-mini, gpt-4o-mini-tts) and same behaviour, but:
- talker() actually gets wired into the graph now (it existed in the notebook
  but was commented out of the Gradio pipeline).
- The `response.executable_ad_data` bug is fixed to `response.content`
  (chatbot_multimodal.py already made this same fix independently).
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
        model="gpt-image-1-mini",
        n=1,
        size="1024x1024",
        response_format="b64_json",
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
"""Support for OpenAI Text to Speech."""

import asyncio
import logging
import time
from io import BytesIO
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import MaxLengthExceeded
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from openai import OpenAI

from .const import (
    TITLE,
    TTS_ENTITY_UNIQUE_ID,
    CONF_API_KEY,
    CONF_TTS_MODEL,
    TTS_MODELS,
    TTS_VOICES,
    CONF_TTS_RESPONSE_FORMAT,
    TTS_RESPONSE_FORMATS,
    CONF_TTS_SPEED,
    DEFAULT_TTS_SPEED,
    SUPPORTED_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4096


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up OpenAI TTS entry."""
    if CONF_API_KEY not in config_entry.data:
        return

    openai_client = await asyncio.to_thread(
        lambda: OpenAI(api_key=config_entry.data[CONF_API_KEY])
    )

    async_add_entities([OpenAITTSEntity(openai_client, config_entry)])


class OpenAITTSEntity(TextToSpeechEntity):
    """The OpenAI TTS entity."""

    _attr_name = TITLE
    _attr_unique_id = TTS_ENTITY_UNIQUE_ID

    def __init__(self, openai_client: OpenAI, config_entry: ConfigEntry):
        """Initialize TTS entity."""
        self.openai_client = openai_client
        self.tts_model = config_entry.data.get(CONF_TTS_MODEL, TTS_MODELS[0])
        self.tts_response_format = config_entry.data.get(
            CONF_TTS_RESPONSE_FORMAT, TTS_RESPONSE_FORMATS[0]
        )
        self.tts_speed = config_entry.data.get(CONF_TTS_SPEED, DEFAULT_TTS_SPEED)

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options like voice."""
        return [ATTR_VOICE]

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return a list of supported voices for a language."""
        return [Voice(voice, voice) for voice in TTS_VOICES]

    @staticmethod
    def __time_delta(start_time: float, end_time: float) -> int:
        """Calculate the time delta in milliseconds."""
        return int((end_time - start_time) * 1000)

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Convert text to speech and return it as bytes."""
        tts_voice: str = options.get(ATTR_VOICE, TTS_VOICES[0])
        tts_voice = tts_voice if tts_voice in TTS_VOICES else TTS_VOICES[0]

        try:
            if len(message) > MAX_MESSAGE_LENGTH:
                raise MaxLengthExceeded

            start_time = time.time()
            receive_time = start_time
            audio_data = BytesIO()

            with self.openai_client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=tts_voice,
                input=message,
                response_format=self.tts_response_format,
                speed=self.tts_speed,
            ) as response:
                receive_time = time.time()
                for chunk in response.iter_bytes(chunk_size=1024):
                    audio_data.write(chunk)

            end_time = time.time()
            _LOGGER.info(
                f"TTS Delay: {OpenAITTSEntity.__time_delta(start_time, receive_time)}ms; Duration: {OpenAITTSEntity.__time_delta(receive_time, end_time)}ms; Total: {OpenAITTSEntity.__time_delta(start_time, end_time)}ms"
            )

            audio_bytes = audio_data.getvalue()

            # The response should contain the audio file content
            return (self.tts_response_format, audio_bytes)
        except MaxLengthExceeded:
            _LOGGER.error("Maximum length of the message exceeded.")
        except Exception as e:
            _LOGGER.error("Unknown Error: %s", e, exc_info=True)

        return (None, None)

"""Support for OpenAI Text to Speech."""

import asyncio
import logging
import struct
import wave
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
    CONF_REALTIME_ENABLED,
    CONF_TTS_USE_REALTIME,
    REALTIME_VOICES,
    SUPPORTED_LANGUAGES,
)
from .realtime_client import get_realtime_client

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

    # Determine if we should use Realtime API for TTS
    realtime_enabled = config_entry.data.get(CONF_REALTIME_ENABLED, False)
    use_realtime_tts = config_entry.data.get(CONF_TTS_USE_REALTIME, False)

    if realtime_enabled and use_realtime_tts:
        async_add_entities([OpenAIRealtimeTTSEntity(hass, config_entry)])
    else:
        openai_client = await asyncio.to_thread(
            lambda: OpenAI(api_key=config_entry.data[CONF_API_KEY])
        )
        async_add_entities([OpenAITTSEntity(openai_client, config_entry)])


class OpenAITTSEntity(TextToSpeechEntity):
    """The OpenAI TTS entity using the standard API."""

    _attr_name = TITLE
    _attr_unique_id = TTS_ENTITY_UNIQUE_ID
    _attr_default_language = "en"
    _attr_supported_languages = SUPPORTED_LANGUAGES
    _attr_supported_options = [ATTR_VOICE]

    def __init__(self, openai_client: OpenAI, config_entry: ConfigEntry):
        """Initialize TTS entity."""
        self.openai_client = openai_client
        self.tts_model = config_entry.data.get(CONF_TTS_MODEL, TTS_MODELS[0])
        self.tts_response_format = config_entry.data.get(
            CONF_TTS_RESPONSE_FORMAT, TTS_RESPONSE_FORMATS[0]
        )
        self.tts_speed = config_entry.data.get(CONF_TTS_SPEED, DEFAULT_TTS_SPEED)

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return a list of supported voices for a language."""
        return [Voice(voice, voice) for voice in TTS_VOICES]

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Convert text to speech and return it as bytes."""
        tts_voice: str = options.get(ATTR_VOICE, TTS_VOICES[0])
        tts_voice = tts_voice if tts_voice in TTS_VOICES else TTS_VOICES[0]

        try:
            if len(message) > MAX_MESSAGE_LENGTH:
                raise MaxLengthExceeded(message, "message", MAX_MESSAGE_LENGTH)

            # Generate TTS audio
            audio_bytes = await asyncio.to_thread(
                self._generate_tts_audio, message, tts_voice
            )

            # The response should contain the audio file content
            return (self.tts_response_format, audio_bytes)
        except MaxLengthExceeded:
            _LOGGER.error("Maximum length of the message exceeded.")
        except Exception as e:
            _LOGGER.error("Unknown Error: %s", e, exc_info=True)

        return (None, None)

    def _generate_tts_audio(self, message: str, tts_voice: str) -> bytes:
        """Generate TTS audio."""
        audio_data = BytesIO()

        with self.openai_client.audio.speech.with_streaming_response.create(
            model=self.tts_model,
            voice=tts_voice,
            input=message,
            response_format=self.tts_response_format,
            speed=self.tts_speed,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                audio_data.write(chunk)

        return audio_data.getvalue()


class OpenAIRealtimeTTSEntity(TextToSpeechEntity):
    """The OpenAI TTS entity using the Realtime API for lower latency."""

    _attr_name = f"{TITLE} (Realtime)"
    _attr_unique_id = f"{TTS_ENTITY_UNIQUE_ID}-realtime"
    _attr_default_language = "en"
    _attr_supported_languages = SUPPORTED_LANGUAGES
    _attr_supported_options = [ATTR_VOICE]

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize Realtime TTS entity."""
        self.hass = hass
        self._config_entry = config_entry

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return a list of supported voices for a language."""
        # Realtime API has a different set of voices
        return [Voice(voice, voice) for voice in REALTIME_VOICES]

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Convert text to speech using Realtime API."""
        tts_voice: str = options.get(ATTR_VOICE, REALTIME_VOICES[0])
        tts_voice = tts_voice if tts_voice in REALTIME_VOICES else REALTIME_VOICES[0]

        try:
            if len(message) > MAX_MESSAGE_LENGTH:
                raise MaxLengthExceeded(message, "message", MAX_MESSAGE_LENGTH)

            client = get_realtime_client(self.hass, self._config_entry)
            if client is None:
                _LOGGER.error("Realtime client not available for TTS")
                return (None, None)

            # Use the Realtime API for TTS
            response = await client.text_to_speech(message, voice=tts_voice)

            if response.error:
                _LOGGER.error("Realtime TTS error: %s", response.error)
                return (None, None)

            if not response.audio_data:
                _LOGGER.error("No audio data received from Realtime API")
                return (None, None)

            # Convert PCM16 24kHz mono to WAV format
            wav_data = self._pcm16_to_wav(response.audio_data, sample_rate=24000)

            return ("wav", wav_data)

        except MaxLengthExceeded:
            _LOGGER.error("Maximum length of the message exceeded.")
        except Exception as e:
            _LOGGER.error("Realtime TTS Error: %s", e, exc_info=True)

        return (None, None)

    def _pcm16_to_wav(
        self, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1
    ) -> bytes:
        """Convert raw PCM16 audio data to WAV format."""
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()

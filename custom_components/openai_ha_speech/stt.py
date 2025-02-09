"""Support for OpenAI Speech to Text."""

import asyncio
from collections.abc import AsyncIterable
from io import BytesIO
import logging
import time
import wave

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from openai import OpenAI, NotGiven, NOT_GIVEN

from .const import (
    TITLE,
    STT_ENTITY_UNIQUE_ID,
    CONF_API_KEY,
    CONF_STT_MODEL,
    STT_MODELS,
    CONF_STT_LANGUAGE,
    CONF_STT_TEMPERATURE,
    DEFAULT_STT_TEMPERATURE,
    SUPPORTED_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up OpenAI STT entry."""
    if CONF_API_KEY not in config_entry.data:
        return

    openai_client = await asyncio.to_thread(
        lambda: OpenAI(api_key=config_entry.data[CONF_API_KEY])
    )
    async_add_entities([OpenAISTTEntity(openai_client, config_entry)])


class OpenAISTTEntity(SpeechToTextEntity):
    """The OpenAI STT entity."""

    _attr_name = TITLE
    _attr_unique_id = STT_ENTITY_UNIQUE_ID

    def __init__(self, openai_client: OpenAI, config_entry: ConfigEntry):
        """Initialize STT entity."""
        self.openai_client = openai_client
        self.stt_model = config_entry.data.get(CONF_STT_MODEL, STT_MODELS[0])
        self.stt_language: str | NotGiven = config_entry.data.get(
            CONF_STT_LANGUAGE, NOT_GIVEN
        )
        self.stt_temperature = config_entry.data.get(
            CONF_STT_TEMPERATURE, DEFAULT_STT_TEMPERATURE
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    @staticmethod
    def __time_delta(start_time: float, end_time: float) -> int:
        """Calculate the time delta in milliseconds."""
        return int((end_time - start_time) * 1000)

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Convert speech into text."""
        try:
            start_time = time.time()

            # Convert the audio stream into bytes
            audio_bytes: bytes = b""
            async for chunk in stream:
                audio_bytes += chunk

            audio_stream = BytesIO()
            with wave.open(audio_stream, "wb") as wf:
                wf.setnchannels(metadata.channel)
                wf.setsampwidth(metadata.bit_rate // 8)
                wf.setframerate(metadata.sample_rate)
                wf.writeframes(audio_bytes)

            audio_file = ("stt_audio.wav", audio_stream, "audio/wav")

            translate_time = time.time()

            transcription = await asyncio.to_thread(
                lambda: self.openai_client.audio.transcriptions.create(
                    model=self.stt_model,
                    language=self.stt_language,
                    temperature=self.stt_temperature,
                    response_format="json",
                    file=audio_file,
                )
            )

            end_time = time.time()
            _LOGGER.info(
                f"STT Delay: {OpenAISTTEntity.__time_delta(start_time, translate_time)}ms; Duration: {OpenAISTTEntity.__time_delta(translate_time, end_time)}ms; Total: {OpenAISTTEntity.__time_delta(start_time, end_time)}ms"
            )

            # The response should contain the transcription
            return SpeechResult(transcription.text, SpeechResultState.SUCCESS)
        except Exception as e:
            _LOGGER.error("Unknown Error: %s", e, exc_info=True)

        return SpeechResult(None, SpeechResultState.ERROR)

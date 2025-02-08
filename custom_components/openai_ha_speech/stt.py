"""Support for OpenAI Speech to Text."""

from collections.abc import AsyncIterable
import logging
import time
from io import BytesIO
from typing import Any

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

    async_add_entities([OpenAISTTEntity(config_entry)])


class OpenAISTTEntity(SpeechToTextEntity):
    """The OpenAI STT entity."""

    def __init__(self, config_entry: ConfigEntry):
        """Initialize STT entity."""
        self.openai_client = OpenAI(api_key=config_entry.data[CONF_API_KEY])
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
            receive_time = start_time

            transcription = self.openai_client.audio.transcriptions.create(
                model=self.stt_model,
                language=self.stt_language,
                temperature=self.stt_temperature,
                file=stream,
            )

            end_time = time.time()
            _LOGGER.info(
                f"STT Delay: {self.__time_delta(start_time, receive_time)}ms; Duration: {self.__time_delta(receive_time, end_time)}ms; Total: {self.__time_delta(start_time, end_time)}ms"
            )

            # The response should contain the transcription
            return SpeechResult(transcription.text, SpeechResultState.SUCCESS)
        except Exception as e:
            _LOGGER.error("Unknown Error: %s", e, exc_info=True)

        return SpeechResult(None, SpeechResultState.ERROR)

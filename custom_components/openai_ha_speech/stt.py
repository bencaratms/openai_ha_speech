"""Support for OpenAI Speech to Text."""

import asyncio
from collections.abc import AsyncIterable
from io import BytesIO
import logging
import wave
import audioop

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

from openai import OpenAI

from .const import (
    TITLE,
    STT_ENTITY_UNIQUE_ID,
    CONF_API_KEY,
    CONF_STT_MODEL,
    STT_MODELS,
    CONF_STT_LANGUAGE,
    CONF_STT_TEMPERATURE,
    DEFAULT_STT_TEMPERATURE,
    CONF_REALTIME_ENABLED,
    CONF_STT_USE_REALTIME,
    SUPPORTED_LANGUAGES,
)
from .realtime_client import get_realtime_client

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up OpenAI STT entry."""
    if CONF_API_KEY not in config_entry.data:
        return

    # Determine if we should use Realtime API for STT
    realtime_enabled = config_entry.data.get(CONF_REALTIME_ENABLED, False)
    use_realtime_stt = config_entry.data.get(CONF_STT_USE_REALTIME, False)

    if realtime_enabled and use_realtime_stt:
        async_add_entities([OpenAIRealtimeSTTEntity(hass, config_entry)])
    else:
        openai_client = await asyncio.to_thread(
            lambda: OpenAI(api_key=config_entry.data[CONF_API_KEY])
        )
        async_add_entities([OpenAISTTEntity(openai_client, config_entry)])


class OpenAISTTEntity(SpeechToTextEntity):
    """The OpenAI STT entity using the standard Whisper API."""

    _attr_name = TITLE
    _attr_unique_id = STT_ENTITY_UNIQUE_ID

    def __init__(self, openai_client: OpenAI, config_entry: ConfigEntry):
        """Initialize STT entity."""
        self.openai_client = openai_client
        self.stt_model = config_entry.data.get(CONF_STT_MODEL, STT_MODELS[0])
        self.stt_language: str | None = config_entry.data.get(CONF_STT_LANGUAGE, None)
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

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Convert speech into text."""
        try:
            # Convert the audio stream into bytes
            audio_bytes: bytes = b""
            async for chunk in stream:
                audio_bytes += chunk

            # Convert the audio bytes into a WAV file
            audio_stream = BytesIO()
            with wave.open(audio_stream, "wb") as wf:
                wf.setnchannels(metadata.channel)
                wf.setsampwidth(metadata.bit_rate // 8)
                wf.setframerate(metadata.sample_rate)
                wf.writeframes(audio_bytes)
            audio_file = ("stt_audio.wav", audio_stream, "audio/wav")

            # Transcribe the audio file
            transcribe_kwargs: dict = {
                "model": self.stt_model,
                "temperature": self.stt_temperature,
                "response_format": "json",
                "file": audio_file,
            }
            if self.stt_language is not None:
                transcribe_kwargs["language"] = self.stt_language

            transcription = await asyncio.to_thread(
                lambda: self.openai_client.audio.transcriptions.create(
                    **transcribe_kwargs
                )
            )

            # The response should contain the transcription
            return SpeechResult(transcription.text, SpeechResultState.SUCCESS)
        except Exception as e:
            _LOGGER.error("Unknown Error: %s", e, exc_info=True)

        return SpeechResult(None, SpeechResultState.ERROR)


class OpenAIRealtimeSTTEntity(SpeechToTextEntity):
    """The OpenAI STT entity using the Realtime API for lower latency."""

    _attr_name = f"{TITLE} (Realtime)"
    _attr_unique_id = f"{STT_ENTITY_UNIQUE_ID}-realtime"

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize Realtime STT entity."""
        self.hass = hass
        self._config_entry = config_entry

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        # Realtime API supports 24kHz, but we'll resample from 16kHz
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Convert speech into text using Realtime API."""
        try:
            # Collect all audio data
            audio_bytes: bytes = b""
            async for chunk in stream:
                audio_bytes += chunk

            if not audio_bytes:
                _LOGGER.warning("No audio data received")
                return SpeechResult(None, SpeechResultState.ERROR)

            # Resample from 16kHz to 24kHz for Realtime API
            # The Realtime API expects 24kHz PCM16 mono audio
            resampled_audio = await asyncio.to_thread(
                self._resample_audio,
                audio_bytes,
                metadata.sample_rate,
                24000,
            )

            client = get_realtime_client(self.hass, self._config_entry)
            if client is None:
                _LOGGER.error("Realtime client not available for STT")
                return SpeechResult(None, SpeechResultState.ERROR)

            # Use the Realtime API for transcription
            response = await client.speech_to_text(resampled_audio, sample_rate=24000)

            if response.error:
                _LOGGER.error("Realtime STT error: %s", response.error)
                return SpeechResult(None, SpeechResultState.ERROR)

            transcript = response.transcript or response.text
            if not transcript:
                _LOGGER.warning("No transcription received from Realtime API")
                return SpeechResult("", SpeechResultState.SUCCESS)

            return SpeechResult(transcript, SpeechResultState.SUCCESS)

        except Exception as e:
            _LOGGER.error("Realtime STT Error: %s", e, exc_info=True)

        return SpeechResult(None, SpeechResultState.ERROR)

    def _resample_audio(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample PCM16 audio data to a different sample rate."""
        if from_rate == to_rate:
            return audio_data

        # Use audioop for simple resampling
        # Convert to mono 16-bit if needed
        resampled, _ = audioop.ratecv(
            audio_data,
            2,  # 16-bit = 2 bytes per sample
            1,  # mono
            from_rate,
            to_rate,
            None,
        )
        return resampled

"""Standard OpenAI API client for TTS and STT."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

_LOGGER = logging.getLogger(__name__)


@dataclass
class StandardConfig:
    """Configuration for standard API client."""

    api_key: str
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    tts_speed: float = 1.0
    stt_model: str = "whisper-1"
    stt_language: str | None = None
    stt_temperature: float = 0.0

    @classmethod
    def from_dict(cls, api_key: str, config: dict) -> "StandardConfig":
        """Create config from dictionary."""
        tts_config = config.get("tts", {})
        stt_config = config.get("stt", {})
        return cls(
            api_key=api_key,
            tts_model=tts_config.get("model", "tts-1"),
            tts_voice=tts_config.get("voice", "alloy"),
            tts_response_format=tts_config.get("response_format", "mp3"),
            tts_speed=tts_config.get("speed", 1.0),
            stt_model=stt_config.get("model", "whisper-1"),
            stt_language=stt_config.get("language"),
            stt_temperature=stt_config.get("temperature", 0.0),
        )


@dataclass
class StandardResponse:
    """Response from standard API operations."""

    audio_data: bytes = b""
    text: str = ""
    format: str = ""
    error: str | None = None


class OpenAIStandardClient:
    """Client for standard OpenAI TTS/STT APIs."""

    def __init__(self, config: StandardConfig) -> None:
        """Initialize the client."""
        self.config = config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.config.api_key)
        return self._client

    async def text_to_speech(
        self, text: str, voice: str | None = None
    ) -> StandardResponse:
        """Convert text to speech using standard TTS API."""
        try:
            client = self._get_client()
            voice = voice or self.config.tts_voice

            def _generate():
                audio_data = BytesIO()
                with client.audio.speech.with_streaming_response.create(
                    model=self.config.tts_model,
                    voice=voice,
                    input=text,
                    response_format=self.config.tts_response_format,
                    speed=self.config.tts_speed,
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=1024):
                        audio_data.write(chunk)
                return audio_data.getvalue()

            audio_bytes = await asyncio.to_thread(_generate)

            return StandardResponse(
                audio_data=audio_bytes,
                format=self.config.tts_response_format,
            )

        except Exception as e:
            _LOGGER.exception("TTS error: %s", e)
            return StandardResponse(error=str(e))

    async def speech_to_text(
        self, audio_data: bytes, audio_format: str = "wav"
    ) -> StandardResponse:
        """Convert speech to text using standard Whisper API."""
        try:
            client = self._get_client()

            audio_file = (
                f"audio.{audio_format}",
                BytesIO(audio_data),
                f"audio/{audio_format}",
            )

            transcribe_kwargs: dict[str, Any] = {
                "model": self.config.stt_model,
                "temperature": self.config.stt_temperature,
                "response_format": "json",
                "file": audio_file,
            }
            if self.config.stt_language:
                transcribe_kwargs["language"] = self.config.stt_language

            def _transcribe():
                return client.audio.transcriptions.create(**transcribe_kwargs)

            transcription = await asyncio.to_thread(_transcribe)

            return StandardResponse(text=transcription.text)

        except Exception as e:
            _LOGGER.exception("STT error: %s", e)
            return StandardResponse(error=str(e))

    async def chat(
        self, text: str, history: list[ChatCompletionMessageParam] | None = None
    ) -> StandardResponse:
        """Send a chat message using standard Chat API."""
        try:
            client = self._get_client()

            messages: list[ChatCompletionMessageParam] = list(history) if history else []
            messages.append({"role": "user", "content": text})

            def _chat():
                return client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )

            response = await asyncio.to_thread(_chat)
            response_text = response.choices[0].message.content or ""

            return StandardResponse(text=response_text)

        except Exception as e:
            _LOGGER.exception("Chat error: %s", e)
            return StandardResponse(error=str(e))

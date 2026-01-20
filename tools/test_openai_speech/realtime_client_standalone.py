"""Standalone OpenAI Realtime API WebSocket Client.

This is a standalone version without Home Assistant dependencies.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import aiohttp

_LOGGER = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Default values
DEFAULT_MODEL = "gpt-4o-realtime-preview"
DEFAULT_VOICE = "alloy"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant. Be concise and friendly."
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 4096
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_SILENCE_DURATION = 500
DEFAULT_PREFIX_PADDING = 300


class SessionModality(Enum):
    """Session modality types."""

    TEXT_ONLY = ["text"]
    AUDIO_ONLY = ["audio"]
    TEXT_AND_AUDIO = ["text", "audio"]


@dataclass
class RealtimeConfig:
    """Configuration for the Realtime API client."""

    api_key: str
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    instructions: str = DEFAULT_INSTRUCTIONS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    silence_duration_ms: int = DEFAULT_SILENCE_DURATION
    prefix_padding_ms: int = DEFAULT_PREFIX_PADDING
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"

    @classmethod
    def from_dict(cls, api_key: str, config: dict) -> "RealtimeConfig":
        """Create config from a dictionary."""
        return cls(
            api_key=api_key,
            model=config.get("model", DEFAULT_MODEL),
            voice=config.get("voice", DEFAULT_VOICE),
            instructions=config.get("instructions", DEFAULT_INSTRUCTIONS),
            temperature=config.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=config.get("max_tokens", DEFAULT_MAX_TOKENS),
            vad_threshold=config.get("vad_threshold", DEFAULT_VAD_THRESHOLD),
            silence_duration_ms=config.get(
                "silence_duration_ms", DEFAULT_SILENCE_DURATION
            ),
            prefix_padding_ms=config.get("prefix_padding_ms", DEFAULT_PREFIX_PADDING),
        )


@dataclass
class RealtimeResponse:
    """Response from a Realtime API operation."""

    text: str = ""
    audio_data: bytes = b""
    transcript: str = ""
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class OpenAIRealtimeClient:
    """Standalone WebSocket client for OpenAI Realtime API."""

    def __init__(self, config: RealtimeConfig) -> None:
        """Initialize the Realtime API client."""
        self.config = config

        # Connection state
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_connected = False
        self._connection_lock = asyncio.Lock()
        self._last_activity: float = 0

        # Session state
        self._current_modalities: list[str] = []
        self._conversation_history: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        """Return whether the client is connected."""
        return self._is_connected and self._ws is not None and not self._ws.closed

    async def connect(
        self,
        modalities: SessionModality = SessionModality.TEXT_ONLY,
        enable_vad: bool = False,
    ) -> bool:
        """Establish WebSocket connection to the Realtime API."""
        async with self._connection_lock:
            if self.is_connected:
                if self._current_modalities == modalities.value:
                    return True
                await self._update_session(modalities, enable_vad)
                return True

            await self._cleanup_connection()

            url = f"{OPENAI_REALTIME_URL}?model={self.config.model}"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

            try:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    url,
                    headers=headers,
                    heartbeat=30.0,
                )

                # Wait for session.created event
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "session.created":
                            _LOGGER.debug("Realtime session created")
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        _LOGGER.error("WebSocket error: %s", self._ws.exception())
                        await self._cleanup_connection()
                        return False

                await self._update_session(modalities, enable_vad)

                self._is_connected = True
                self._last_activity = time.time()
                _LOGGER.info("Connected to OpenAI Realtime API")
                return True

            except aiohttp.ClientError as err:
                _LOGGER.error("Failed to connect to OpenAI Realtime API: %s", err)
                await self._cleanup_connection()
                return False
            except Exception as err:
                _LOGGER.exception(
                    "Unexpected error connecting to Realtime API: %s", err
                )
                await self._cleanup_connection()
                return False

    async def _update_session(
        self, modalities: SessionModality, enable_vad: bool
    ) -> None:
        """Update session configuration."""
        if not self._ws:
            return

        self._current_modalities = modalities.value

        session_config: dict[str, Any] = {
            "modalities": modalities.value,
            "instructions": self.config.instructions,
            "voice": self.config.voice,
            "temperature": self.config.temperature,
            "max_response_output_tokens": self.config.max_tokens,
            "input_audio_format": self.config.input_audio_format,
            "output_audio_format": self.config.output_audio_format,
        }

        if enable_vad and "audio" in modalities.value:
            session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": self.config.vad_threshold,
                "silence_duration_ms": self.config.silence_duration_ms,
                "prefix_padding_ms": self.config.prefix_padding_ms,
            }
        else:
            session_config["turn_detection"] = None

        await self._ws.send_json(
            {
                "type": "session.update",
                "session": session_config,
            }
        )
        _LOGGER.debug("Session configuration updated: modalities=%s", modalities.value)

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket and session resources."""
        self._is_connected = False
        self._current_modalities = []
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def disconnect(self) -> None:
        """Disconnect from the Realtime API."""
        async with self._connection_lock:
            await self._cleanup_connection()
            self._conversation_history.clear()
            _LOGGER.info("Disconnected from OpenAI Realtime API")

    async def send_text(self, text: str) -> RealtimeResponse:
        """Send a text message and get a text response."""
        if not await self.connect(SessionModality.TEXT_ONLY):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._last_activity = time.time()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            await ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
            _LOGGER.debug("User message sent: %s", text)

            await ws.send_json({"type": "response.create"})

            response = RealtimeResponse()
            response_parts: list[str] = []

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")
                    response.events.append(data)

                    if event_type == "response.text.delta":
                        response_parts.append(data.get("delta", ""))
                    elif event_type == "response.text.done":
                        response.text = data.get("text", "".join(response_parts))
                    elif event_type == "response.done":
                        break
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        _LOGGER.error("Realtime API error: %s", error_msg)
                        await self.disconnect()
                        return RealtimeResponse(error=error_msg)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error: %s", ws.exception())
                    await self.disconnect()
                    return RealtimeResponse(error="Connection error")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    await self.disconnect()
                    return RealtimeResponse(error="Connection lost")

            if not response.text and response_parts:
                response.text = "".join(response_parts)

            self._conversation_history.append({"role": "user", "content": text})
            self._conversation_history.append(
                {"role": "assistant", "content": response.text}
            )
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

            return response

        except Exception as err:
            _LOGGER.exception("Error in send_text: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    async def text_to_speech(
        self, text: str, voice: str | None = None
    ) -> RealtimeResponse:
        """Convert text to speech using the Realtime API."""
        if not await self.connect(SessionModality.TEXT_AND_AUDIO):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._last_activity = time.time()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            if voice and voice != self.config.voice:
                await ws.send_json(
                    {
                        "type": "session.update",
                        "session": {"voice": voice},
                    }
                )

            await ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )

            await ws.send_json({"type": "response.create"})

            response = RealtimeResponse()
            audio_chunks: list[bytes] = []

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")
                    response.events.append(data)

                    if event_type == "response.audio.delta":
                        audio_b64 = data.get("delta", "")
                        if audio_b64:
                            audio_chunks.append(base64.b64decode(audio_b64))
                    elif event_type == "response.audio_transcript.done":
                        response.transcript = data.get("transcript", "")
                    elif event_type == "response.text.done":
                        response.text = data.get("text", "")
                    elif event_type == "response.done":
                        break
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        _LOGGER.error("TTS error: %s", error_msg)
                        return RealtimeResponse(error=error_msg)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    return RealtimeResponse(error="Connection error")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    return RealtimeResponse(error="Connection lost")

            response.audio_data = b"".join(audio_chunks)

            if voice and voice != self.config.voice:
                await ws.send_json(
                    {
                        "type": "session.update",
                        "session": {"voice": self.config.voice},
                    }
                )

            return response

        except Exception as err:
            _LOGGER.exception("Error in text_to_speech: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    async def speech_to_text(self, audio_data: bytes) -> RealtimeResponse:
        """Convert speech to text using the Realtime API."""
        if not await self.connect(SessionModality.TEXT_AND_AUDIO, enable_vad=False):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._last_activity = time.time()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            # Send audio in chunks
            chunk_size = 16384
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                audio_b64 = base64.b64encode(chunk).decode("utf-8")
                await ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )

            await ws.send_json({"type": "input_audio_buffer.commit"})
            await ws.send_json({"type": "response.create"})

            response = RealtimeResponse()
            transcript_parts: list[str] = []

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")
                    response.events.append(data)

                    if (
                        event_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        response.transcript = data.get("transcript", "")
                    elif event_type == "response.audio_transcript.delta":
                        transcript_parts.append(data.get("delta", ""))
                    elif event_type == "response.audio_transcript.done":
                        if not response.transcript:
                            response.transcript = data.get(
                                "transcript", "".join(transcript_parts)
                            )
                    elif event_type == "response.text.delta":
                        transcript_parts.append(data.get("delta", ""))
                    elif event_type == "response.text.done":
                        if not response.transcript:
                            response.transcript = data.get(
                                "text", "".join(transcript_parts)
                            )
                    elif event_type == "response.done":
                        break
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        _LOGGER.error("STT error: %s", error_msg)
                        return RealtimeResponse(error=error_msg)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    return RealtimeResponse(error="Connection error")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    return RealtimeResponse(error="Connection lost")

            if not response.transcript and transcript_parts:
                response.transcript = "".join(transcript_parts)

            return response

        except Exception as err:
            _LOGGER.exception("Error in speech_to_text: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    async def voice_conversation(self, audio_data: bytes) -> RealtimeResponse:
        """Send audio and get audio response (full voice conversation turn)."""
        if not await self.connect(SessionModality.TEXT_AND_AUDIO, enable_vad=False):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._last_activity = time.time()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            # Send audio
            chunk_size = 16384
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                audio_b64 = base64.b64encode(chunk).decode("utf-8")
                await ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )

            await ws.send_json({"type": "input_audio_buffer.commit"})
            await ws.send_json({"type": "response.create"})

            response = RealtimeResponse()
            audio_chunks: list[bytes] = []
            transcript_parts: list[str] = []

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")
                    response.events.append(data)

                    if event_type == "response.audio.delta":
                        audio_b64 = data.get("delta", "")
                        if audio_b64:
                            audio_chunks.append(base64.b64decode(audio_b64))
                    elif (
                        event_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        response.transcript = data.get("transcript", "")
                    elif event_type == "response.audio_transcript.done":
                        response.text = data.get("transcript", "")
                    elif event_type == "response.text.done":
                        if not response.text:
                            response.text = data.get("text", "")
                    elif event_type == "response.done":
                        break
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        return RealtimeResponse(error=error_msg)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    return RealtimeResponse(error="Connection error")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    return RealtimeResponse(error="Connection lost")

            response.audio_data = b"".join(audio_chunks)
            return response

        except Exception as err:
            _LOGGER.exception("Error in voice_conversation: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

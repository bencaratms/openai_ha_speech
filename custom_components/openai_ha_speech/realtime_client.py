"""OpenAI Realtime API WebSocket Client.

This module provides a shared WebSocket client for the OpenAI Realtime API
that can be used by conversation, TTS, and STT components.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

import aiohttp

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

from .const import (
    CONF_API_KEY,
    CONF_REALTIME_MODEL,
    CONF_REALTIME_VOICE,
    CONF_REALTIME_INSTRUCTIONS,
    CONF_REALTIME_TEMPERATURE,
    CONF_REALTIME_MAX_TOKENS,
    CONF_REALTIME_IDLE_TIMEOUT,
    CONF_REALTIME_VAD_THRESHOLD,
    CONF_REALTIME_SILENCE_DURATION,
    CONF_REALTIME_PREFIX_PADDING,
    REALTIME_MODELS,
    REALTIME_VOICES,
    DEFAULT_REALTIME_INSTRUCTIONS,
    DEFAULT_REALTIME_TEMPERATURE,
    DEFAULT_REALTIME_MAX_TOKENS,
    DEFAULT_REALTIME_IDLE_TIMEOUT,
    DEFAULT_REALTIME_VAD_THRESHOLD,
    DEFAULT_REALTIME_SILENCE_DURATION,
    DEFAULT_REALTIME_PREFIX_PADDING,
)

_LOGGER = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class SessionModality(Enum):
    """Session modality types for OpenAI Realtime API."""

    TEXT_ONLY = ["text"]
    TEXT_AND_AUDIO = ["text", "audio"]


@dataclass
class RealtimeConfig:
    """Configuration for the Realtime API client."""

    api_key: str
    model: str = REALTIME_MODELS[0]
    voice: str = REALTIME_VOICES[0]
    instructions: str = DEFAULT_REALTIME_INSTRUCTIONS
    temperature: float = DEFAULT_REALTIME_TEMPERATURE
    max_tokens: int = DEFAULT_REALTIME_MAX_TOKENS
    idle_timeout: int = DEFAULT_REALTIME_IDLE_TIMEOUT
    vad_threshold: float = DEFAULT_REALTIME_VAD_THRESHOLD
    silence_duration_ms: int = DEFAULT_REALTIME_SILENCE_DURATION
    prefix_padding_ms: int = DEFAULT_REALTIME_PREFIX_PADDING
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"

    @classmethod
    def from_config_entry(cls, config_entry: ConfigEntry) -> "RealtimeConfig":
        """Create config from a Home Assistant config entry."""
        data = config_entry.data
        return cls(
            api_key=data[CONF_API_KEY],
            model=data.get(CONF_REALTIME_MODEL, REALTIME_MODELS[0]),
            voice=data.get(CONF_REALTIME_VOICE, REALTIME_VOICES[0]),
            instructions=data.get(
                CONF_REALTIME_INSTRUCTIONS, DEFAULT_REALTIME_INSTRUCTIONS
            ),
            temperature=data.get(
                CONF_REALTIME_TEMPERATURE, DEFAULT_REALTIME_TEMPERATURE
            ),
            max_tokens=data.get(CONF_REALTIME_MAX_TOKENS, DEFAULT_REALTIME_MAX_TOKENS),
            idle_timeout=data.get(
                CONF_REALTIME_IDLE_TIMEOUT, DEFAULT_REALTIME_IDLE_TIMEOUT
            ),
            vad_threshold=data.get(
                CONF_REALTIME_VAD_THRESHOLD, DEFAULT_REALTIME_VAD_THRESHOLD
            ),
            silence_duration_ms=data.get(
                CONF_REALTIME_SILENCE_DURATION, DEFAULT_REALTIME_SILENCE_DURATION
            ),
            prefix_padding_ms=data.get(
                CONF_REALTIME_PREFIX_PADDING, DEFAULT_REALTIME_PREFIX_PADDING
            ),
        )

    @classmethod
    def from_dict(cls, api_key: str, data: dict[str, Any]) -> "RealtimeConfig":
        """Create config from a dictionary (for standalone usage).

        Args:
            api_key: The OpenAI API key.
            data: Configuration dictionary with optional keys matching field names.

        Returns:
            A RealtimeConfig instance.
        """
        return cls(
            api_key=api_key,
            model=data.get("model", REALTIME_MODELS[0]),
            voice=data.get("voice", REALTIME_VOICES[0]),
            instructions=data.get("instructions", DEFAULT_REALTIME_INSTRUCTIONS),
            temperature=data.get("temperature", DEFAULT_REALTIME_TEMPERATURE),
            max_tokens=data.get("max_tokens", DEFAULT_REALTIME_MAX_TOKENS),
            idle_timeout=data.get("idle_timeout", DEFAULT_REALTIME_IDLE_TIMEOUT),
            vad_threshold=data.get("vad_threshold", DEFAULT_REALTIME_VAD_THRESHOLD),
            silence_duration_ms=data.get(
                "silence_duration_ms", DEFAULT_REALTIME_SILENCE_DURATION
            ),
            prefix_padding_ms=data.get(
                "prefix_padding_ms", DEFAULT_REALTIME_PREFIX_PADDING
            ),
            input_audio_format=data.get("input_audio_format", "pcm16"),
            output_audio_format=data.get("output_audio_format", "pcm16"),
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
    """Shared WebSocket client for OpenAI Realtime API.

    This client manages persistent WebSocket connections and provides
    methods for text, TTS, and STT operations.
    """

    def __init__(
        self,
        config: RealtimeConfig,
        hass: HomeAssistant | None = None,
    ) -> None:
        """Initialize the Realtime API client.

        Args:
            config: The realtime configuration.
            hass: Optional Home Assistant instance. If provided, uses hass.async_create_task
                  for better task management. If None, uses asyncio.create_task.
        """
        self.hass = hass
        self.config = config

        # Connection state
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_connected = False
        self._connection_lock = asyncio.Lock()
        self._last_activity: float = 0
        self._idle_timeout_task: asyncio.Task | None = None

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
        """Establish WebSocket connection to the Realtime API.

        Args:
            modalities: The session modalities to enable.
            enable_vad: Whether to enable server-side voice activity detection.

        Returns:
            True if connection was successful.
        """
        async with self._connection_lock:
            # Check if already connected with same modalities
            if self.is_connected:
                if self._current_modalities == modalities.value:
                    self._reset_idle_timeout()
                    return True
                # Need to reconfigure session
                await self._update_session(modalities, enable_vad)
                return True

            # Clean up any stale connection
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

                # Configure session
                await self._update_session(modalities, enable_vad)

                self._is_connected = True
                self._reset_idle_timeout()
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
            # Cancel idle timeout task
            if self._idle_timeout_task:
                self._idle_timeout_task.cancel()
                try:
                    await self._idle_timeout_task
                except asyncio.CancelledError:
                    pass
                self._idle_timeout_task = None

            await self._cleanup_connection()
            self._conversation_history.clear()
            _LOGGER.info("Disconnected from OpenAI Realtime API")

    def _reset_idle_timeout(self) -> None:
        """Reset the idle timeout timer."""
        self._last_activity = time.time()

        # Cancel existing timeout task
        if self._idle_timeout_task:
            self._idle_timeout_task.cancel()

        # Start new timeout task (use hass.async_create_task if available for HA integration)
        if self.hass is not None:
            self._idle_timeout_task = self.hass.async_create_task(
                self._idle_timeout_handler()
            )
        else:
            self._idle_timeout_task = asyncio.create_task(self._idle_timeout_handler())

    async def _idle_timeout_handler(self) -> None:
        """Handle idle timeout - disconnect after inactivity."""
        try:
            await asyncio.sleep(self.config.idle_timeout)
            _LOGGER.debug("Idle timeout reached, disconnecting")
            await self.disconnect()
        except asyncio.CancelledError:
            pass

    # =========================================================================
    # Text Operations
    # =========================================================================

    async def send_text(self, text: str) -> RealtimeResponse:
        """Send a text message and get a text response.

        Args:
            text: The text message to send.

        Returns:
            RealtimeResponse with the text response.
        """
        if not await self.connect(SessionModality.TEXT_ONLY):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._reset_idle_timeout()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            # Send user message
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

            # Request response
            await ws.send_json({"type": "response.create"})

            # Collect response
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

            # Update conversation history
            self._conversation_history.append({"role": "user", "content": text})
            self._conversation_history.append(
                {"role": "assistant", "content": response.text}
            )
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

            self._reset_idle_timeout()
            return response

        except Exception as err:
            _LOGGER.exception("Error in send_text: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    # =========================================================================
    # Text-to-Speech Operations
    # =========================================================================

    async def text_to_speech(
        self, text: str, voice: str | None = None
    ) -> RealtimeResponse:
        """Convert text to speech using the Realtime API.

        Args:
            text: The text to convert to speech.
            voice: Optional voice override.

        Returns:
            RealtimeResponse with audio_data containing PCM16 audio.
        """
        if not await self.connect(SessionModality.TEXT_AND_AUDIO):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._reset_idle_timeout()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            # If voice override requested, update session temporarily
            if voice and voice != self.config.voice:
                await ws.send_json(
                    {
                        "type": "session.update",
                        "session": {"voice": voice},
                    }
                )

            # Send text for TTS
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

            # Request audio response
            await ws.send_json({"type": "response.create"})

            # Collect response
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
                    elif event_type == "response.audio.done":
                        pass
                    elif event_type == "response.audio_transcript.delta":
                        pass
                    elif event_type == "response.audio_transcript.done":
                        response.transcript = data.get("transcript", "")
                    elif event_type == "response.text.delta":
                        pass  # May receive text too
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

            # Restore original voice if it was changed
            if voice and voice != self.config.voice:
                await ws.send_json(
                    {
                        "type": "session.update",
                        "session": {"voice": self.config.voice},
                    }
                )

            self._reset_idle_timeout()
            return response

        except Exception as err:
            _LOGGER.exception("Error in text_to_speech: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    # =========================================================================
    # Speech-to-Text Operations
    # =========================================================================

    async def speech_to_text(
        self,
        audio_data: bytes,
        sample_rate: int = 24000,
    ) -> RealtimeResponse:
        """Convert speech to text using the Realtime API.

        Args:
            audio_data: PCM16 audio data.
            sample_rate: Sample rate of the audio (default 24kHz for Realtime API).

        Returns:
            RealtimeResponse with transcript containing the transcription.
        """
        if not await self.connect(SessionModality.TEXT_AND_AUDIO, enable_vad=False):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._reset_idle_timeout()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            # Send audio in chunks (16KB at a time)
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

            # Commit audio buffer
            await ws.send_json({"type": "input_audio_buffer.commit"})

            # Request transcription response
            await ws.send_json({"type": "response.create"})

            # Collect response
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
                        _LOGGER.debug("Transcription received: %s", response.transcript)
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

            self._reset_idle_timeout()
            return response

        except Exception as err:
            _LOGGER.exception("Error in speech_to_text: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    async def voice_conversation(
        self,
        audio_data: bytes,
        sample_rate: int = 24000,
    ) -> RealtimeResponse:
        """Send audio and get both text and audio response.

        This method sends audio input and retrieves a full conversational response
        including both the transcription of the input and the assistant's response
        with text and audio.

        Args:
            audio_data: PCM16 audio data.
            sample_rate: Sample rate of the audio (default 24kHz for Realtime API).

        Returns:
            RealtimeResponse with transcript (input), text (response), and audio_data.
        """
        if not await self.connect(SessionModality.TEXT_AND_AUDIO, enable_vad=False):
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        self._reset_idle_timeout()

        ws = self._ws
        if ws is None:
            return RealtimeResponse(error="Failed to connect to OpenAI service")

        try:
            # Send audio in chunks (16KB at a time)
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

            # Commit audio buffer and request response
            await ws.send_json({"type": "input_audio_buffer.commit"})
            await ws.send_json({"type": "response.create"})

            # Collect response
            response = RealtimeResponse()
            transcript_parts: list[str] = []
            text_parts: list[str] = []
            audio_chunks: list[bytes] = []

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")
                    response.events.append(data)

                    # Input transcription
                    if (
                        event_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        response.transcript = data.get("transcript", "")
                        _LOGGER.debug("Input transcription: %s", response.transcript)

                    # Response text
                    elif event_type == "response.text.delta":
                        text_parts.append(data.get("delta", ""))
                    elif event_type == "response.text.done":
                        response.text = data.get("text", "".join(text_parts))

                    # Response audio transcript (fallback for text)
                    elif event_type == "response.audio_transcript.delta":
                        if not text_parts:  # Only if no text response
                            transcript_parts.append(data.get("delta", ""))
                    elif event_type == "response.audio_transcript.done":
                        if not response.text:
                            response.text = data.get(
                                "transcript", "".join(transcript_parts)
                            )

                    # Response audio
                    elif event_type == "response.audio.delta":
                        audio_b64 = data.get("delta", "")
                        if audio_b64:
                            audio_chunks.append(base64.b64decode(audio_b64))

                    elif event_type == "response.done":
                        break

                    elif event_type == "error":
                        error_msg = data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        _LOGGER.error("Voice conversation error: %s", error_msg)
                        return RealtimeResponse(error=error_msg)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    return RealtimeResponse(error="Connection error")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    return RealtimeResponse(error="Connection lost")

            # Combine audio chunks
            if audio_chunks:
                response.audio_data = b"".join(audio_chunks)

            # Fallback for text
            if not response.text and transcript_parts:
                response.text = "".join(transcript_parts)

            self._reset_idle_timeout()
            return response

        except Exception as err:
            _LOGGER.exception("Error in voice_conversation: %s", err)
            await self.disconnect()
            return RealtimeResponse(error=str(err))

    # =========================================================================
    # Streaming Audio Operations (for real-time bidirectional audio)
    # =========================================================================

    async def start_audio_stream(self, enable_vad: bool = True) -> bool:
        """Start an audio streaming session.

        Args:
            enable_vad: Whether to enable server-side voice activity detection.

        Returns:
            True if the stream was started successfully.
        """
        return await self.connect(SessionModality.TEXT_AND_AUDIO, enable_vad=enable_vad)

    async def send_audio_chunk(self, audio_data: bytes) -> None:
        """Send an audio chunk during streaming.

        Args:
            audio_data: PCM16 audio data chunk.
        """
        if not self.is_connected or not self._ws:
            _LOGGER.warning("Cannot send audio: not connected")
            return

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self._ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )
        self._reset_idle_timeout()

    async def commit_audio_and_respond(self) -> None:
        """Commit the audio buffer and request a response."""
        if not self.is_connected or not self._ws:
            return

        await self._ws.send_json({"type": "input_audio_buffer.commit"})
        await self._ws.send_json({"type": "response.create"})
        self._reset_idle_timeout()

    async def receive_events(self) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields events from the Realtime API.

        Use this for streaming audio responses.

        Yields:
            Event dictionaries from the API.
        """
        if not self.is_connected or not self._ws:
            return

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                yield data

                if data.get("type") == "response.done":
                    break
                elif data.get("type") == "error":
                    break

            elif msg.type == aiohttp.WSMsgType.ERROR:
                _LOGGER.error("WebSocket error in receive_events")
                break

            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                break

        self._reset_idle_timeout()


def get_realtime_client(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> OpenAIRealtimeClient | None:
    """Get or create the shared Realtime API client for a config entry.

    Args:
        hass: Home Assistant instance.
        config_entry: The config entry.

    Returns:
        The shared OpenAIRealtimeClient instance, or None if not available.
    """
    from .const import DOMAIN

    # Store client in hass.data
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    entry_data = hass.data[DOMAIN].setdefault(config_entry.entry_id, {})

    if "realtime_client" not in entry_data:
        config = RealtimeConfig.from_config_entry(config_entry)
        entry_data["realtime_client"] = OpenAIRealtimeClient(config, hass)

    return entry_data["realtime_client"]


async def cleanup_realtime_client(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> None:
    """Clean up the Realtime API client for a config entry."""
    from .const import DOMAIN

    if DOMAIN not in hass.data:
        return

    entry_data = hass.data[DOMAIN].get(config_entry.entry_id, {})
    client = entry_data.pop("realtime_client", None)
    if client:
        await client.disconnect()

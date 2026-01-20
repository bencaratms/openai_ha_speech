"""OpenAI Realtime API conversation agent for Home Assistant."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import intent

from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONVERSATION_ENTITY_UNIQUE_ID,
    CONF_REALTIME_ENABLED,
    CONF_REALTIME_MODEL,
    REALTIME_MODELS,
    CONF_REALTIME_VOICE,
    REALTIME_VOICES,
    CONF_REALTIME_INSTRUCTIONS,
    DEFAULT_REALTIME_INSTRUCTIONS,
    CONF_REALTIME_TEMPERATURE,
    DEFAULT_REALTIME_TEMPERATURE,
    CONF_REALTIME_MAX_TOKENS,
    DEFAULT_REALTIME_MAX_TOKENS,
    SUPPORTED_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up OpenAI Realtime conversation entity from a config entry."""
    # Only set up if realtime is enabled
    if not config_entry.data.get(CONF_REALTIME_ENABLED, False):
        _LOGGER.debug("Realtime API is not enabled, skipping conversation entity setup")
        return

    async_add_entities(
        [OpenAIRealtimeConversationEntity(config_entry)],
        update_before_add=False,
    )


class OpenAIRealtimeConversationEntity(conversation.ConversationEntity):
    """OpenAI Realtime API conversation entity."""

    _attr_has_entity_name = True
    _attr_name = "OpenAI Realtime"

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the conversation entity."""
        self._config_entry = config_entry
        self._attr_unique_id = CONVERSATION_ENTITY_UNIQUE_ID
        self._conversation_history: list[dict[str, Any]] = []

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def _api_key(self) -> str:
        """Return the API key."""
        return self._config_entry.data[CONF_API_KEY]

    @property
    def _model(self) -> str:
        """Return the model to use."""
        return self._config_entry.data.get(CONF_REALTIME_MODEL, REALTIME_MODELS[0])

    @property
    def _voice(self) -> str:
        """Return the voice to use."""
        return self._config_entry.data.get(CONF_REALTIME_VOICE, REALTIME_VOICES[0])

    @property
    def _instructions(self) -> str:
        """Return the system instructions."""
        return self._config_entry.data.get(
            CONF_REALTIME_INSTRUCTIONS, DEFAULT_REALTIME_INSTRUCTIONS
        )

    @property
    def _temperature(self) -> float:
        """Return the temperature setting."""
        return self._config_entry.data.get(
            CONF_REALTIME_TEMPERATURE, DEFAULT_REALTIME_TEMPERATURE
        )

    @property
    def _max_tokens(self) -> int:
        """Return the max tokens setting."""
        return self._config_entry.data.get(
            CONF_REALTIME_MAX_TOKENS, DEFAULT_REALTIME_MAX_TOKENS
        )

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence and return a response.

        This method handles text-based conversations using the Realtime API.
        For full audio streaming, a separate audio pipeline handler is needed.
        """
        response_text = await self._send_text_message(user_input.text)

        # Create intent response
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )

    async def _send_text_message(self, text: str) -> str:
        """Send a text message to the Realtime API and get a response."""
        url = f"{OPENAI_REALTIME_URL}?model={self._model}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        response_text = ""
        response_complete = asyncio.Event()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    url,
                    headers=headers,
                    heartbeat=30.0,
                ) as ws:
                    # Wait for session.created event
                    session_created = False
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "session.created":
                                session_created = True
                                _LOGGER.debug("Realtime session created")
                                break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            _LOGGER.error("WebSocket error: %s", ws.exception())
                            return "Sorry, I encountered a connection error."

                    if not session_created:
                        return "Sorry, I couldn't establish a session."

                    # Update session configuration
                    session_update = {
                        "type": "session.update",
                        "session": {
                            "modalities": ["text"],
                            "instructions": self._instructions,
                            "voice": self._voice,
                            "temperature": self._temperature,
                            "max_response_output_tokens": self._max_tokens,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                        },
                    }
                    await ws.send_json(session_update)
                    _LOGGER.debug("Session update sent")

                    # Add conversation history to context
                    for hist_item in self._conversation_history[-10:]:
                        conversation_item = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": hist_item["role"],
                                "content": [
                                    (
                                        {
                                            "type": "input_text",
                                            "text": hist_item["content"],
                                        }
                                        if hist_item["role"] == "user"
                                        else {
                                            "type": "text",
                                            "text": hist_item["content"],
                                        }
                                    )
                                ],
                            },
                        }
                        await ws.send_json(conversation_item)

                    # Send the user's message
                    user_message = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": text}],
                        },
                    }
                    await ws.send_json(user_message)
                    _LOGGER.debug("User message sent: %s", text)

                    # Request a response
                    await ws.send_json({"type": "response.create"})
                    _LOGGER.debug("Response requested")

                    # Collect response
                    response_parts: list[str] = []

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            event_type = data.get("type", "")

                            if event_type == "response.text.delta":
                                delta = data.get("delta", "")
                                response_parts.append(delta)

                            elif event_type == "response.text.done":
                                response_text = data.get(
                                    "text", "".join(response_parts)
                                )
                                _LOGGER.debug("Response text complete")

                            elif event_type == "response.done":
                                _LOGGER.debug("Response complete")
                                break

                            elif event_type == "error":
                                error_msg = data.get("error", {}).get(
                                    "message", "Unknown error"
                                )
                                _LOGGER.error("Realtime API error: %s", error_msg)
                                return f"Sorry, an error occurred: {error_msg}"

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            _LOGGER.error("WebSocket error: %s", ws.exception())
                            return "Sorry, I encountered a connection error."

                    if not response_text and response_parts:
                        response_text = "".join(response_parts)

                    # Store in conversation history
                    self._conversation_history.append({"role": "user", "content": text})
                    self._conversation_history.append(
                        {"role": "assistant", "content": response_text}
                    )

                    # Keep history manageable
                    if len(self._conversation_history) > 20:
                        self._conversation_history = self._conversation_history[-20:]

        except aiohttp.ClientError as err:
            _LOGGER.error("Failed to connect to OpenAI Realtime API: %s", err)
            return "Sorry, I couldn't connect to the OpenAI service."
        except asyncio.TimeoutError:
            _LOGGER.error("Timeout connecting to OpenAI Realtime API")
            return "Sorry, the request timed out."
        except Exception as err:
            _LOGGER.exception("Unexpected error in Realtime API: %s", err)
            return "Sorry, an unexpected error occurred."

        return response_text or "I'm sorry, I couldn't generate a response."


class OpenAIRealtimeAudioHandler:
    """Handler for real-time audio streaming with OpenAI Realtime API.

    This class manages WebSocket connections for audio streaming,
    supporting voice activity detection (VAD) and bidirectional audio.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        instructions: str,
        temperature: float,
        max_tokens: int,
        vad_threshold: float = 0.5,
        silence_duration_ms: int = 500,
        prefix_padding_ms: int = 300,
    ) -> None:
        """Initialize the audio handler."""
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._vad_threshold = vad_threshold
        self._silence_duration_ms = silence_duration_ms
        self._prefix_padding_ms = prefix_padding_ms
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._is_connected = False
        self._audio_buffer: list[bytes] = []
        self._response_audio_buffer: list[bytes] = []

    async def connect(self) -> bool:
        """Establish WebSocket connection to the Realtime API."""
        url = f"{OPENAI_REALTIME_URL}?model={self._model}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(
                url,
                headers=headers,
                heartbeat=30.0,
            )

            # Wait for session.created
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "session.created":
                        _LOGGER.debug("Audio session created")
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error during connect")
                    return False

            # Configure session for audio
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self._instructions,
                    "voice": self._voice,
                    "temperature": self._temperature,
                    "max_response_output_tokens": self._max_tokens,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self._vad_threshold,
                        "silence_duration_ms": self._silence_duration_ms,
                        "prefix_padding_ms": self._prefix_padding_ms,
                    },
                },
            }
            await self._ws.send_json(session_update)
            self._is_connected = True
            return True

        except Exception as err:
            _LOGGER.error("Failed to connect audio handler: %s", err)
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._is_connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session:
            await self._session.close()
            self._session = None

    async def send_audio_chunk(self, audio_data: bytes) -> None:
        """Send an audio chunk to the Realtime API.

        Args:
            audio_data: PCM16 audio data at 24kHz sample rate.
        """
        if not self._is_connected or not self._ws:
            _LOGGER.warning("Cannot send audio: not connected")
            return

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        await self._ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )

    async def commit_audio(self) -> None:
        """Commit the audio buffer to trigger processing."""
        if not self._is_connected or not self._ws:
            return

        await self._ws.send_json({"type": "input_audio_buffer.commit"})
        await self._ws.send_json({"type": "response.create"})

    async def receive_events(self) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields events from the Realtime API.

        Yields:
            Event dictionaries from the API including audio chunks,
            transcriptions, and response completions.
        """
        if not self._is_connected or not self._ws:
            return

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                yield data

                # Handle audio response chunks
                if data.get("type") == "response.audio.delta":
                    audio_b64 = data.get("delta", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        self._response_audio_buffer.append(audio_bytes)

                elif data.get("type") == "response.audio.done":
                    # Audio response complete
                    pass

                elif data.get("type") == "response.done":
                    # Full response complete
                    break

            elif msg.type == aiohttp.WSMsgType.ERROR:
                _LOGGER.error("WebSocket error in receive_events")
                break

    def get_response_audio(self) -> bytes:
        """Get the accumulated response audio data."""
        audio = b"".join(self._response_audio_buffer)
        self._response_audio_buffer.clear()
        return audio

    @property
    def is_connected(self) -> bool:
        """Return whether the handler is connected."""
        return self._is_connected


# Type hint for async iterator
from typing import AsyncIterator

"""OpenAI Realtime API conversation agent for Home Assistant."""

from __future__ import annotations

import logging

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import intent

from .const import (
    CONF_REALTIME_ENABLED,
    CONVERSATION_ENTITY_UNIQUE_ID,
    SUPPORTED_LANGUAGES,
)
from .realtime_client import get_realtime_client, OpenAIRealtimeClient

_LOGGER = logging.getLogger(__name__)


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
        [OpenAIRealtimeConversationEntity(hass, config_entry)],
        update_before_add=False,
    )


class OpenAIRealtimeConversationEntity(conversation.ConversationEntity):
    """OpenAI Realtime API conversation entity using shared client."""

    _attr_has_entity_name = True
    _attr_name = "OpenAI Realtime"

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the conversation entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._attr_unique_id = CONVERSATION_ENTITY_UNIQUE_ID
        self._client: OpenAIRealtimeClient | None = None

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    def _get_client(self) -> OpenAIRealtimeClient | None:
        """Get the shared Realtime client."""
        if self._client is None:
            self._client = get_realtime_client(self.hass, self._config_entry)
        return self._client

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal from Home Assistant."""
        # Client cleanup is handled by __init__.py unload
        pass

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence and return a response."""
        client = self._get_client()
        if client is None:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(
                "Sorry, the OpenAI Realtime service is not available."
            )
            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        response = await client.send_text(user_input.text)

        if response.error:
            response_text = f"Sorry, an error occurred: {response.error}"
        else:
            response_text = (
                response.text or "I'm sorry, I couldn't generate a response."
            )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )

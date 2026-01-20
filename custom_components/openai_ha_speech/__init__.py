"""OpenAI Speech API integration."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_REALTIME_ENABLED

PLATFORMS: list[str] = [Platform.TTS, Platform.STT]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up entities."""
    platforms = list(PLATFORMS)

    # Add conversation platform if realtime is enabled
    if entry.data.get(CONF_REALTIME_ENABLED, False):
        platforms.append(Platform.CONVERSATION)

    await hass.config_entries.async_forward_entry_setups(entry, platforms)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    platforms = list(PLATFORMS)

    if entry.data.get(CONF_REALTIME_ENABLED, False):
        platforms.append(Platform.CONVERSATION)

    return await hass.config_entries.async_unload_platforms(entry, platforms)

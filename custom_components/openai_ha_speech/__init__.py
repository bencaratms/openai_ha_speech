"""OpenAI Speech API integration."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_REALTIME_ENABLED
from .realtime_client import cleanup_realtime_client

PLATFORMS: list[str] = [Platform.TTS, Platform.STT]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up entities."""
    # Initialize domain data storage
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    if entry.entry_id not in hass.data[DOMAIN]:
        hass.data[DOMAIN][entry.entry_id] = {}

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

    # Clean up the realtime client
    await cleanup_realtime_client(hass, entry)

    # Clean up domain data
    if DOMAIN in hass.data:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)

    return await hass.config_entries.async_unload_platforms(entry, platforms)

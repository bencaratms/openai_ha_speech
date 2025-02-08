"""OpenAI Speech API integration."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

PLATFORMS: list[str] = [Platform.TTS, Platform.STT]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up entities."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

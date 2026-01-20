"""Config flow for the OpenAI Speech API integration."""

from __future__ import annotations
from typing import Any
import voluptuous as vol
import logging

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult
from homeassistant.helpers.selector import (
    selector,
    NumberSelector,
    NumberSelectorConfig,
)
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    TITLE,
    CONF_API_KEY,
    CONF_TTS_MODEL,
    TTS_MODELS,
    CONF_TTS_RESPONSE_FORMAT,
    TTS_RESPONSE_FORMATS,
    CONF_TTS_SPEED,
    DEFAULT_TTS_SPEED,
    CONF_TTS_INSTRUCTIONS,
    CONF_STT_MODEL,
    STT_MODELS,
    CONF_STT_LANGUAGE,
    CONF_STT_TEMPERATURE,
    DEFAULT_STT_TEMPERATURE,
    CONF_REALTIME_ENABLED,
    CONF_REALTIME_MODEL,
    REALTIME_MODELS,
    CONF_REALTIME_VOICE,
    REALTIME_VOICES,
    CONF_REALTIME_INSTRUCTIONS,
    DEFAULT_REALTIME_INSTRUCTIONS,
    CONF_REALTIME_TEMPERATURE,
    DEFAULT_REALTIME_TEMPERATURE,
    CONF_REALTIME_IDLE_TIMEOUT,
    DEFAULT_REALTIME_IDLE_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)


class OpenAISpeechConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Speech API integration."""

    VERSION = 1
    MINOR_VERSION = 0

    STEP_USER_DATA_SCHEMA = vol.Schema(
        {
            vol.Required(CONF_API_KEY): str,
            vol.Required(CONF_TTS_MODEL, default=TTS_MODELS[0]): selector(
                {
                    "select": {
                        "options": TTS_MODELS,
                        "mode": "dropdown",
                    }
                }
            ),
            vol.Required(
                CONF_TTS_RESPONSE_FORMAT, default=TTS_RESPONSE_FORMATS[0]
            ): selector(
                {
                    "select": {
                        "options": TTS_RESPONSE_FORMATS,
                        "mode": "dropdown",
                    }
                }
            ),
            vol.Optional(CONF_TTS_SPEED, default=DEFAULT_TTS_SPEED): NumberSelector(
                NumberSelectorConfig(min=0.25, max=4.0, step=0.25)
            ),
            vol.Optional(CONF_TTS_INSTRUCTIONS): str,
            vol.Required(CONF_STT_MODEL, default=STT_MODELS[0]): selector(
                {
                    "select": {
                        "options": STT_MODELS,
                        "mode": "dropdown",
                        "sort": True,
                    }
                }
            ),
            vol.Optional(CONF_STT_LANGUAGE): str,
            vol.Optional(
                CONF_STT_TEMPERATURE, default=DEFAULT_STT_TEMPERATURE
            ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.1)),
            # Realtime API options
            vol.Optional(CONF_REALTIME_ENABLED, default=False): bool,
            vol.Optional(CONF_REALTIME_MODEL, default=REALTIME_MODELS[0]): selector(
                {
                    "select": {
                        "options": REALTIME_MODELS,
                        "mode": "dropdown",
                    }
                }
            ),
            vol.Optional(CONF_REALTIME_VOICE, default=REALTIME_VOICES[0]): selector(
                {
                    "select": {
                        "options": REALTIME_VOICES,
                        "mode": "dropdown",
                    }
                }
            ),
            vol.Optional(
                CONF_REALTIME_INSTRUCTIONS, default=DEFAULT_REALTIME_INSTRUCTIONS
            ): str,
            vol.Optional(
                CONF_REALTIME_TEMPERATURE, default=DEFAULT_REALTIME_TEMPERATURE
            ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.2, step=0.1)),
            vol.Optional(
                CONF_REALTIME_IDLE_TIMEOUT, default=DEFAULT_REALTIME_IDLE_TIMEOUT
            ): NumberSelector(NumberSelectorConfig(min=10, max=300, step=10, unit_of_measurement="seconds")),
        }
    )

    @staticmethod
    def __validate_user_input(user_input: dict[str, Any]):
        """Validate user input fields."""
        if user_input.get(CONF_API_KEY) is None:
            raise ValueError("API key is required")
        if user_input.get(CONF_TTS_MODEL) is None:
            raise ValueError("Text-to-speech model is required")
        if user_input.get(CONF_TTS_RESPONSE_FORMAT) is None:
            raise ValueError("Response format is required")
        if user_input.get(CONF_STT_MODEL) is None:
            raise ValueError("Speach-to-text model is required")

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle a flow initialized by the user."""
        errors: dict[str, Any] = {}
        if user_input is not None:
            try:
                OpenAISpeechConfigFlow.__validate_user_input(user_input)

                # Return the validated data.
                return self.async_create_entry(
                    title=TITLE,
                    data=user_input,
                )
            except ValueError as e:
                _LOGGER.exception(str(e))
                errors["base"] = str(e)

        # Show the form, either unfilled or with errors from a previous attempt.
        return self.async_show_form(
            step_id="user",
            data_schema=self.STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders=user_input,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of the integration."""
        errors: dict[str, Any] = {}
        if user_input is not None:
            try:
                OpenAISpeechConfigFlow.__validate_user_input(user_input)

                # Return the validated data.
                return self.async_update_reload_and_abort(
                    self._get_reconfigure_entry(),
                    data_updates=user_input,
                )
            except HomeAssistantError as e:
                _LOGGER.exception(str(e))
                errors["base"] = str(e)
            except ValueError as e:
                _LOGGER.exception(str(e))
                errors["base"] = str(e)

        # Show the form, either unfilled or with errors from a previous attempt.
        return self.async_show_form(
            step_id="user",
            data_schema=self.STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders=user_input,
        )

    async def async_migrate_entry(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Migrate old entry."""
        _LOGGER.debug(
            "Migrating configuration from version %s.%s.",
            config_entry.version,
            config_entry.minor_version,
        )

        if config_entry.version > self.VERSION or (
            config_entry.version == self.VERSION
            and config_entry.minor_version > self.MINOR_VERSION
        ):
            # This means the user has downgraded from a future version
            _LOGGER.debug(
                "Downgrade from configuration version %s.%s is not supported.",
                config_entry.version,
                config_entry.minor_version,
            )
            return False

        # if config_entry.version == 1:
        #     new_data = {**config_entry.data}
        #     if config_entry.minor_version < 2:
        #         TODO: modify Config Entry data with changes in version 1.2
        #         pass

        #     hass.config_entries.async_update_entry(
        #        config_entry, data=new_data, minor_version=2, version=1
        #     )

        _LOGGER.debug(
            "Migration to configuration version %s.%s successful.",
            config_entry.version,
            config_entry.minor_version,
        )

        return True

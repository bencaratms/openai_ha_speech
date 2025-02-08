"""Config flow for the OpenAI Speech API integration."""

from __future__ import annotations
from typing import Any
import voluptuous as vol
import logging

from homeassistant import data_entry_flow
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.helpers.selector import selector
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    TITLE_FORMAT,
    CONF_API_KEY,
    CONF_TTS_MODEL,
    TTS_MODELS,
    CONF_TTS_VOICE,
    TTS_VOICES,
    CONF_TTS_RESPONSE_FORMAT,
    TTS_RESPONSE_FORMATS,
    CONF_TTS_SPEED,
    CONF_STT_MODEL,
    STT_MODELS,
    CONF_STT_LANGUAGE,
    CONF_STT_TEMPERATURE,
)

_LOGGER = logging.getLogger(__name__)


class OpenAISpeechConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Speech API integration."""

    VERSION = 1

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
            vol.Required(CONF_TTS_VOICE, default=TTS_VOICES[0]): selector(
                {
                    "select": {
                        "options": TTS_VOICES,
                        "mode": "dropdown",
                        "sort": True,
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
            vol.Optional(CONF_TTS_SPEED, default=1.0): vol.Coerce(float),
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
            vol.Optional(CONF_STT_TEMPERATURE, default=0.0): vol.Coerce(float),
        }
    )

    def __generate_unique_id(user_input: dict) -> str:
        """Generate a unique id from user input."""
        return f"openai_ha_speech_{user_input[CONF_TTS_VOICE]}_{user_input[CONF_TTS_MODEL]}"

    def __validate_user_input(user_input: dict):
        """Validate user input fields."""
        if user_input.get(CONF_API_KEY) is None:
            raise ValueError("API key is required")
        if user_input.get(CONF_TTS_MODEL) is None:
            raise ValueError("Text-to-speech model is required")
        if user_input.get(CONF_TTS_VOICE) is None:
            raise ValueError("Voice is required")
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
                self.__validate_user_input(user_input)

                await self.async_set_unique_id(self.__generate_unique_id(user_input))
                self._abort_if_unique_id_configured()

                # Return the validated data.
                return self.async_create_entry(
                    title=TITLE_FORMAT.format(
                        user_input[CONF_TTS_VOICE], user_input[CONF_TTS_MODEL]
                    ),
                    data=user_input,
                )
            except data_entry_flow.AbortFlow:
                return self.async_abort(reason="already_configured")
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

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of the integration."""
        errors: dict[str, Any] = {}
        if user_input is not None:
            try:
                self.__validate_user_input(user_input)

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

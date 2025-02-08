""" Constants for OpenAI Speech API integration."""

DOMAIN = "openai_ha_speech"
TITLE_FORMAT = "OpenAI Speech API ({0}, {1})"

CONF_API_KEY = "api_key"

CONF_TTS_MODEL = "tts_model"
TTS_MODELS = ["tts-1", "tts-1-hd"]
CONF_TTS_VOICE = "tts_voice"
TTS_VOICES = [
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
]
CONF_TTS_RESPONSE_FORMAT = "tts_response_format"
TTS_RESPONSE_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
CONF_TTS_SPEED = "tts_speed"

CONF_STT_MODEL = "stt_model"
STT_MODELS = ["whisper-1"]
CONF_STT_LANGUAGE = "stt_language"
CONF_STT_TEMPERATURE = "stt_temperature"

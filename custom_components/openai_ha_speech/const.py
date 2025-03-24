"""Constants for OpenAI Speech API integration."""

DOMAIN = "openai_ha_speech"
TITLE = "OpenAI Speech API"
STT_ENTITY_UNIQUE_ID = "openai-ha-speech-stt"
TTS_ENTITY_UNIQUE_ID = "openai-ha-speech-tts"

CONF_API_KEY = "api_key"

CONF_TTS_MODEL = "tts_model"
TTS_MODELS = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
TTS_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]
CONF_TTS_RESPONSE_FORMAT = "tts_response_format"
TTS_RESPONSE_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
CONF_TTS_SPEED = "tts_speed"
DEFAULT_TTS_SPEED: float = 1.0
CONF_TTS_INSTRUCTIONS = "tts_instructions"

CONF_STT_MODEL = "stt_model"
STT_MODELS = ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
CONF_STT_LANGUAGE = "stt_language"
CONF_STT_TEMPERATURE = "stt_temperature"
DEFAULT_STT_TEMPERATURE: float = 0.0

SUPPORTED_LANGUAGES = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
]

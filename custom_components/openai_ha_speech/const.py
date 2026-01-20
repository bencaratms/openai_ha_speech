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

# Realtime API constants
CONVERSATION_ENTITY_UNIQUE_ID = "openai-ha-speech-conversation"
CONF_REALTIME_ENABLED = "realtime_enabled"
CONF_REALTIME_MODEL = "realtime_model"
REALTIME_MODELS = ["gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"]
CONF_REALTIME_VOICE = "realtime_voice"
REALTIME_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
CONF_REALTIME_INSTRUCTIONS = "realtime_instructions"
DEFAULT_REALTIME_INSTRUCTIONS = "You are a helpful home assistant. Be concise and helpful."
CONF_REALTIME_TEMPERATURE = "realtime_temperature"
DEFAULT_REALTIME_TEMPERATURE: float = 0.8
CONF_REALTIME_MAX_TOKENS = "realtime_max_tokens"
DEFAULT_REALTIME_MAX_TOKENS: int = 4096
CONF_REALTIME_VAD_THRESHOLD = "realtime_vad_threshold"
DEFAULT_REALTIME_VAD_THRESHOLD: float = 0.5
CONF_REALTIME_SILENCE_DURATION = "realtime_silence_duration"
DEFAULT_REALTIME_SILENCE_DURATION: int = 500
CONF_REALTIME_PREFIX_PADDING = "realtime_prefix_padding"
DEFAULT_REALTIME_PREFIX_PADDING: int = 300

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

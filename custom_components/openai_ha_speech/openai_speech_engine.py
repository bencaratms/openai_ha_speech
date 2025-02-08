from __future__ import annotations
from io import BytesIO
from openai import OpenAI
import logging
import time
from .const import TTS_MODELS, TTS_VOICES, TTS_RESPONSE_FORMATS, STT_MODELS

_LOGGER = logging.getLogger(__name__)


class OpenAISpeechEngine:

    def __init__(
        self,
        api_key: str,
        tts_model: str,
        tts_voice: str,
        tts_response_format: str,
        tts_speed: float,
        stt_model: str,
        stt_language: str,
        stt_temperature: float,
    ):
        if tts_model not in TTS_MODELS:
            raise ValueError(f"Invalid TTS model: {tts_model}")
        if tts_voice not in TTS_VOICES:
            raise ValueError(f"Invalid TTS voice: {tts_voice}")
        if tts_response_format not in TTS_RESPONSE_FORMATS:
            raise ValueError(f"Invalid TTS response format: {tts_response_format}")
        if stt_model not in STT_MODELS:
            raise ValueError(f"Invalid STT model: {stt_model}")

        self.client = OpenAI(api_key=api_key)
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_response_format = tts_response_format
        self.tts_speed = tts_speed
        self.stt_model = stt_model
        self.stt_language = stt_language
        self.stt_temperature = stt_temperature

    def __time_delta(start_time, end_time) -> int:
        return int((end_time - start_time) * 1000)

    def tts_create(self, input: str) -> bytes:
        """
        Generates text-to-speech audio data from the given input text.
        Args:
            input (str): The input text to be converted to speech.
        Returns:
            bytes: The generated audio data in bytes.
        Logs:
            Logs the delay, duration, and total time taken for the TTS process.
        Raises:
            Any exceptions raised by the underlying TTS client will propagate.
        """
        start_time = time.time()
        receive_time = start_time
        audio_data = BytesIO()

        with self.client.audio.speech.with_streaming_response.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=input,
            response_format=self.tts_response_format,
            speed=self.tts_speed,
        ) as response:
            receive_time = time.time()
            for chunk in response.iter_bytes(chunk_size=1024):
                audio_data.write(chunk)

        end_time = time.time()
        _LOGGER.info(
            f"Delay: {self.__time_delta(start_time, receive_time)}ms; Duration: {self.__time_delta(receive_time, end_time)}ms; Total: {self.__time_delta(start_time, end_time)}ms"
        )

        return audio_data.getvalue()

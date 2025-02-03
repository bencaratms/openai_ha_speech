from io import BytesIO
from openai import OpenAI
from typing import Literal
import logging
import time

_LOGGER = logging.getLogger(__name__)


class OpenAISpeechEngine:

    def __init__(
        self,
        api_key: str,
        model: Literal["tts-1", "tts-1-hd"],
        voice: Literal[
            "alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"
        ],
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"],
        speed: float,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed

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
            model=self.model,
            voice=self.voice,
            input=input,
            response_format=self.response_format,
            speed=self.speed,
        ) as response:
            receive_time = time.time()
            for chunk in response.iter_bytes(chunk_size=1024):
                audio_data.write(chunk)

        end_time = time.time()
        _LOGGER.info(
            f"Delay: {self.__time_delta(start_time, receive_time)}ms; Duration: {self.__time_delta(receive_time, end_time)}ms; Total: {self.__time_delta(start_time, end_time)}ms"
        )

        return audio_data.getvalue()

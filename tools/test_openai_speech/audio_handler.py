"""Audio recording, playback, and file handling."""

from __future__ import annotations

import asyncio
import audioop
import logging
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Try to import sounddevice, but handle if not available
try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    _LOGGER.warning("sounddevice not available: %s", e)


class AudioHandler:
    """Handles audio recording, playback, resampling, and file operations."""

    def __init__(
        self,
        output_dir: str = "output",
        input_device: int | None = None,
        output_device: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Initialize audio handler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._recorded_data: list[np.ndarray] = []

    @staticmethod
    def list_devices() -> str:
        """List available audio devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return "Audio devices not available (sounddevice not installed)"

        import sounddevice as sd

        return str(sd.query_devices())

    def _generate_filename(self, prefix: str, extension: str) -> Path:
        """Generate a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{prefix}_{timestamp}.{extension}"

    async def record(
        self,
        duration: float | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> bytes:
        """Record audio from microphone.

        Args:
            duration: Recording duration in seconds. If None, records until stop_callback returns True.
            stop_callback: Function that returns True when recording should stop.

        Returns:
            PCM16 audio data at configured sample rate.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "Audio recording not available (sounddevice not installed)"
            )

        import sounddevice as sd

        self._recording = True
        self._recorded_data = []

        def callback(indata, frames, time_info, status):
            if status:
                _LOGGER.warning("Recording status: %s", status)
            if self._recording:
                self._recorded_data.append(indata.copy())

        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                callback=callback,
            ):
                if duration:
                    await asyncio.sleep(duration)
                elif stop_callback:
                    while self._recording and not stop_callback():
                        await asyncio.sleep(0.1)
                else:
                    # Wait for stop() to be called
                    while self._recording:
                        await asyncio.sleep(0.1)
        finally:
            self._recording = False

        if not self._recorded_data:
            return b""

        audio_array = np.concatenate(self._recorded_data)
        audio_bytes = audio_array.tobytes()

        # Save to file
        filepath = self._generate_filename("recording", "wav")
        self.save_wav(audio_bytes, filepath, self.sample_rate, self.channels)
        _LOGGER.info("Saved recording to %s", filepath)

        return audio_bytes

    def stop_recording(self) -> None:
        """Stop ongoing recording."""
        self._recording = False

    async def play(self, audio_data: bytes, sample_rate: int = 24000) -> None:
        """Play audio data through speakers.

        Args:
            audio_data: PCM16 audio data.
            sample_rate: Sample rate of the audio data.
        """
        if not SOUNDDEVICE_AVAILABLE:
            _LOGGER.warning("Audio playback not available (sounddevice not installed)")
            return

        import sounddevice as sd

        # Save to file first
        filepath = self._generate_filename("playback", "wav")
        self.save_wav(audio_data, filepath, sample_rate, 1)
        _LOGGER.info("Saved playback to %s", filepath)

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        try:
            # Play audio (blocking in thread)
            await asyncio.to_thread(
                sd.play,
                audio_array,
                samplerate=sample_rate,
                device=self.output_device,
            )
            await asyncio.to_thread(sd.wait)
        except Exception as e:
            _LOGGER.error("Playback error: %s", e)

    def save_wav(
        self,
        audio_data: bytes,
        filepath: Path,
        sample_rate: int,
        channels: int = 1,
    ) -> Path:
        """Save audio data to WAV file."""
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return filepath

    def load_wav(self, filepath: Path) -> tuple[bytes, int]:
        """Load audio data from WAV file.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        with wave.open(str(filepath), "rb") as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
        return audio_data, sample_rate

    def save_audio(
        self,
        audio_data: bytes,
        format: str,
        prefix: str = "audio",
    ) -> Path:
        """Save audio data to file with appropriate format."""
        filepath = self._generate_filename(prefix, format)
        with open(filepath, "wb") as f:
            f.write(audio_data)
        _LOGGER.info("Saved audio to %s", filepath)
        return filepath

    @staticmethod
    def resample(
        audio_data: bytes,
        from_rate: int,
        to_rate: int,
        channels: int = 1,
    ) -> bytes:
        """Resample PCM16 audio data to a different sample rate."""
        if from_rate == to_rate:
            return audio_data

        resampled, _ = audioop.ratecv(
            audio_data,
            2,  # 16-bit = 2 bytes per sample
            channels,
            from_rate,
            to_rate,
            None,
        )
        return resampled

    @staticmethod
    def pcm16_to_wav(
        pcm_data: bytes,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> bytes:
        """Convert raw PCM16 audio data to WAV format."""
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()

    @staticmethod
    def wav_to_pcm16(wav_data: bytes) -> tuple[bytes, int, int]:
        """Extract PCM16 data from WAV format.

        Returns:
            Tuple of (pcm_data, sample_rate, channels).
        """
        wav_buffer = BytesIO(wav_data)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            pcm_data = wav_file.readframes(wav_file.getnframes())
        return pcm_data, sample_rate, channels

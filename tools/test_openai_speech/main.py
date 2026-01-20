#!/usr/bin/env python3
"""OpenAI Speech Test Application.

A standalone console application for testing STT, TTS, and Conversation
using OpenAI's standard and Realtime APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Setup colored output
try:
    from colorama import init, Fore, Style

    init()
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

    class _ForeType:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = RESET = ""

    class _StyleType:
        BRIGHT = RESET_ALL = ""

    Fore = _ForeType()
    Style = _StyleType()


from audio_handler import AudioHandler
from realtime_client_standalone import OpenAIRealtimeClient, RealtimeConfig
from standard_client import OpenAIStandardClient, StandardConfig
from script_runner import ScriptRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def print_color(text: str, color: str = Fore.WHITE, bright: bool = False) -> None:
    """Print colored text."""
    style = Style.BRIGHT if bright else ""
    print(f"{style}{color}{text}{Style.RESET_ALL}")


def print_header(text: str) -> None:
    """Print a header."""
    print_color(f"\n{'='*60}", Fore.CYAN)
    print_color(f"  {text}", Fore.CYAN, bright=True)
    print_color(f"{'='*60}", Fore.CYAN)


def print_info(label: str, value: str) -> None:
    """Print info line."""
    print(f"  {Fore.YELLOW}{label}:{Style.RESET_ALL} {value}")


def print_success(text: str) -> None:
    """Print success message."""
    print_color(f"✓ {text}", Fore.GREEN)


def print_error(text: str) -> None:
    """Print error message."""
    print_color(f"✗ {text}", Fore.RED)


def print_debug(text: str) -> None:
    """Print debug message."""
    print_color(f"  [DEBUG] {text}", Fore.MAGENTA)


class TestApp:
    """Main test application."""

    def __init__(
        self,
        config: dict,
        api_key: str,
        text_only: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize test application."""
        self.config = config
        self.api_key = api_key
        self.text_only = text_only
        self.debug = debug
        self.mode = config.get("mode", "realtime")

        # Initialize clients
        realtime_config = RealtimeConfig.from_dict(api_key, config.get("realtime", {}))
        self.realtime_client = OpenAIRealtimeClient(realtime_config)

        standard_config = StandardConfig.from_dict(api_key, config)
        self.standard_client = OpenAIStandardClient(standard_config)

        # Initialize audio handler
        audio_config = config.get("audio", {})
        self.audio = AudioHandler(
            output_dir=config.get("output_dir", "output"),
            input_device=audio_config.get("input_device"),
            output_device=audio_config.get("output_device"),
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
        )

        self._chat_history: list[dict] = []
        self._last_response = ""

    async def run_interactive(self) -> None:
        """Run interactive mode."""
        print_header("OpenAI Speech Test Application")
        print_info("Mode", self.mode)
        print_info("Text-only", str(self.text_only))
        print_info("Debug", str(self.debug))
        print()

        self._print_help()

        try:
            while True:
                try:
                    user_input = input(f"\n{Fore.GREEN}>{Style.RESET_ALL} ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ("quit", "exit", "q"):
                    print_info("Goodbye", "Disconnecting...")
                    break
                elif command == "help":
                    self._print_help()
                elif command == "config":
                    self._print_config()
                elif command == "mode":
                    self._set_mode(args)
                elif command == "devices":
                    print(AudioHandler.list_devices())
                elif command == "tts":
                    await self._handle_tts(args)
                elif command == "stt":
                    await self._handle_stt()
                elif command == "chat":
                    await self._handle_chat(args)
                elif command == "voice":
                    await self._handle_voice()
                elif command == "clear":
                    self._chat_history.clear()
                    print_success("Chat history cleared")
                else:
                    print_error(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

        finally:
            await self.realtime_client.disconnect()

    def _print_help(self) -> None:
        """Print help text."""
        print_color("\nAvailable Commands:", Fore.CYAN, bright=True)
        commands = [
            ("tts <text>", "Convert text to speech"),
            ("stt", "Record audio and transcribe"),
            ("chat <text>", "Send text to conversation agent"),
            ("voice", "Voice conversation (record → respond)"),
            ("mode <standard|realtime>", "Switch API mode"),
            ("devices", "List audio devices"),
            ("config", "Show current configuration"),
            ("clear", "Clear chat history"),
            ("help", "Show this help"),
            ("quit", "Exit application"),
        ]
        for cmd, desc in commands:
            print(f"  {Fore.YELLOW}{cmd:30}{Style.RESET_ALL} {desc}")

    def _print_config(self) -> None:
        """Print current configuration."""
        print_header("Current Configuration")
        print_info("Mode", self.mode)
        print_info("Output Directory", self.config.get("output_dir", "output"))

        print_color("\n  TTS Settings:", Fore.CYAN)
        tts = self.config.get("tts", {})
        print_info("    Model", tts.get("model", "tts-1"))
        print_info("    Voice", tts.get("voice", "alloy"))
        print_info("    Format", tts.get("response_format", "mp3"))

        print_color("\n  STT Settings:", Fore.CYAN)
        stt = self.config.get("stt", {})
        print_info("    Model", stt.get("model", "whisper-1"))
        print_info("    Language", str(stt.get("language", "auto")))

        print_color("\n  Realtime Settings:", Fore.CYAN)
        rt = self.config.get("realtime", {})
        print_info("    Model", rt.get("model", "gpt-4o-realtime-preview"))
        print_info("    Voice", rt.get("voice", "alloy"))
        print_info("    Temperature", str(rt.get("temperature", 0.8)))

    def _set_mode(self, mode: str) -> None:
        """Set API mode."""
        if mode in ("standard", "realtime"):
            self.mode = mode
            print_success(f"Mode set to: {mode}")
        else:
            print_error("Mode must be 'standard' or 'realtime'")

    async def _handle_tts(self, text: str) -> None:
        """Handle TTS command."""
        if not text:
            print_error("Please provide text: tts <text>")
            return

        print_info("TTS", f"Converting: {text[:50]}...")

        if self.mode == "realtime":
            response = await self.realtime_client.text_to_speech(text)
            if response.error:
                print_error(f"TTS Error: {response.error}")
                return

            if self.debug:
                print_debug(f"Received {len(response.audio_data)} bytes of audio")
                print_debug(f"Events: {len(response.events)}")

            if response.audio_data:
                # Save audio
                filepath = self.audio.save_audio(
                    self.audio.pcm16_to_wav(response.audio_data, 24000),
                    "wav",
                    "tts_realtime",
                )
                print_success(f"Audio saved: {filepath}")

                # Play audio
                if not self.text_only:
                    print_info("Playing", "audio...")
                    await self.audio.play(response.audio_data, 24000)

        else:  # standard mode
            response = await self.standard_client.text_to_speech(text)
            if response.error:
                print_error(f"TTS Error: {response.error}")
                return

            if self.debug:
                print_debug(f"Received {len(response.audio_data)} bytes of audio")

            # Save audio
            filepath = self.audio.save_audio(
                response.audio_data,
                response.format,
                "tts_standard",
            )
            print_success(f"Audio saved: {filepath}")

            # Note: Standard TTS returns MP3/etc, playback would need different handling
            if not self.text_only:
                print_info("Note", "Standard TTS playback requires audio decoding")

        print_success("TTS complete")

    async def _handle_stt(self) -> None:
        """Handle STT command."""
        if self.text_only:
            print_error("STT not available in text-only mode")
            return

        print_info("STT", "Recording... Press Enter to stop")

        # Start recording in background
        stop_flag = [False]

        def check_stop():
            return stop_flag[0]

        async def wait_for_enter():
            await asyncio.to_thread(input)
            stop_flag[0] = True

        # Run recording and input wait concurrently
        record_task = asyncio.create_task(self.audio.record(stop_callback=check_stop))
        input_task = asyncio.create_task(wait_for_enter())

        # Wait for recording to complete
        done, pending = await asyncio.wait(
            [record_task, input_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.audio.stop_recording()
        audio_data = await record_task if record_task in done else await record_task

        if not audio_data:
            print_error("No audio recorded")
            return

        print_info("Recorded", f"{len(audio_data)} bytes")

        # Transcribe
        if self.mode == "realtime":
            # Resample to 24kHz for Realtime API
            audio_24k = self.audio.resample(audio_data, self.audio.sample_rate, 24000)

            if self.debug:
                print_debug(f"Resampled from {self.audio.sample_rate}Hz to 24000Hz")
                print_debug(f"Audio size: {len(audio_data)} -> {len(audio_24k)} bytes")

            response = await self.realtime_client.speech_to_text(audio_24k)
            if response.error:
                print_error(f"STT Error: {response.error}")
                return

            transcript = response.transcript or response.text
            self._last_response = transcript

            if self.debug:
                print_debug(f"Events: {len(response.events)}")

        else:  # standard mode
            # Convert to WAV
            wav_data = self.audio.pcm16_to_wav(audio_data, self.audio.sample_rate)
            response = await self.standard_client.speech_to_text(wav_data, "wav")
            if response.error:
                print_error(f"STT Error: {response.error}")
                return

            transcript = response.text
            self._last_response = transcript

        print_color(f"\n  Transcription:", Fore.CYAN)
        print_color(f"  {transcript}", Fore.WHITE, bright=True)
        print()

    async def _handle_chat(self, text: str) -> None:
        """Handle chat command."""
        if not text:
            print_error("Please provide text: chat <text>")
            return

        print_info("Chat", f"Sending: {text[:50]}...")

        if self.mode == "realtime":
            response = await self.realtime_client.send_text(text)
            if response.error:
                print_error(f"Chat Error: {response.error}")
                return

            if self.debug:
                print_debug(f"Events: {len(response.events)}")

            self._last_response = response.text

        else:  # standard mode
            response = await self.standard_client.chat(text, self._chat_history.copy())
            if response.error:
                print_error(f"Chat Error: {response.error}")
                return

            self._chat_history.append({"role": "user", "content": text})
            self._chat_history.append({"role": "assistant", "content": response.text})
            self._last_response = response.text

        print_color(f"\n  Response:", Fore.CYAN)
        print_color(f"  {self._last_response}", Fore.WHITE, bright=True)
        print()

    async def _handle_voice(self) -> None:
        """Handle voice conversation command."""
        if self.text_only:
            print_error("Voice mode not available in text-only mode")
            return

        if self.mode != "realtime":
            print_error("Voice conversation only available in realtime mode")
            return

        print_info("Voice", "Recording... Press Enter to stop")

        # Record audio
        stop_flag = [False]

        def check_stop():
            return stop_flag[0]

        async def wait_for_enter():
            await asyncio.to_thread(input)
            stop_flag[0] = True

        record_task = asyncio.create_task(self.audio.record(stop_callback=check_stop))
        input_task = asyncio.create_task(wait_for_enter())

        done, pending = await asyncio.wait(
            [record_task, input_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.audio.stop_recording()
        audio_data = await record_task if record_task in done else await record_task

        if not audio_data:
            print_error("No audio recorded")
            return

        print_info("Recorded", f"{len(audio_data)} bytes")
        print_info("Processing", "Sending to Realtime API...")

        # Resample to 24kHz
        audio_24k = self.audio.resample(audio_data, self.audio.sample_rate, 24000)

        # Send for voice conversation
        response = await self.realtime_client.voice_conversation(audio_24k)

        if response.error:
            print_error(f"Voice Error: {response.error}")
            return

        if self.debug:
            print_debug(f"Events: {len(response.events)}")
            print_debug(f"Transcript: {response.transcript}")
            print_debug(f"Response text: {response.text}")
            print_debug(f"Audio size: {len(response.audio_data)} bytes")

        # Show transcription
        if response.transcript:
            print_color(f"\n  You said:", Fore.YELLOW)
            print_color(f"  {response.transcript}", Fore.WHITE)

        # Show response
        if response.text:
            print_color(f"\n  Response:", Fore.CYAN)
            print_color(f"  {response.text}", Fore.WHITE, bright=True)

        self._last_response = response.text

        # Play response audio
        if response.audio_data:
            print_info("Playing", "response audio...")
            await self.audio.play(response.audio_data, 24000)

        print()

    async def run_script(self, script_path: Path) -> None:
        """Run a test script."""
        print_header(f"Running Script: {script_path.name}")

        runner = ScriptRunner(
            tts_handler=self._script_tts,
            stt_handler=self._script_stt,
            chat_handler=self._script_chat,
            record_handler=self.audio.record,
            play_handler=self.audio.play,
            load_audio_handler=self.audio.load_wav,
        )

        try:
            script = runner.load_script(script_path)
            results = await runner.run(script)
            print(runner.get_summary())
        finally:
            await self.realtime_client.disconnect()

    async def _script_tts(self, text: str):
        """TTS handler for script runner."""
        if self.mode == "realtime":
            response = await self.realtime_client.text_to_speech(text)
            if response.audio_data and not self.text_only:
                await self.audio.play(response.audio_data, 24000)
            return response
        else:
            return await self.standard_client.text_to_speech(text)

    async def _script_stt(self, audio_data: bytes | None, duration: float | None):
        """STT handler for script runner."""
        if audio_data is None:
            if duration:
                audio_data = await self.audio.record(duration=duration)
            else:
                print_error("No audio data and no duration specified")
                return None

        if self.mode == "realtime":
            audio_24k = self.audio.resample(audio_data, self.audio.sample_rate, 24000)
            return await self.realtime_client.speech_to_text(audio_24k)
        else:
            wav_data = self.audio.pcm16_to_wav(audio_data, self.audio.sample_rate)
            return await self.standard_client.speech_to_text(wav_data, "wav")

    async def _script_chat(self, text: str):
        """Chat handler for script runner."""
        if self.mode == "realtime":
            return await self.realtime_client.send_text(text)
        else:
            return await self.standard_client.chat(text, self._chat_history.copy())


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenAI Speech Test Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode with default config
  python main.py --config my_config.json  # Use custom config
  python main.py --text-only              # No audio device access
  python main.py --script test.json       # Run automated script
  python main.py --debug                  # Enable debug output
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.json"),
        help="Path to configuration JSON file (default: config.json)",
    )
    parser.add_argument(
        "--script",
        "-s",
        type=Path,
        help="Path to test script JSON file for automated testing",
    )
    parser.add_argument(
        "--text-only",
        "-t",
        action="store_true",
        help="Text-only mode (no audio device access)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY environment variable not set!")
        print()
        print("To set your API key:")
        print()
        print("  Option 1: Create a .env file in this directory:")
        print("    OPENAI_API_KEY=sk-your-api-key-here")
        print()
        print("  Option 2: Set environment variable:")
        print("    Windows:  set OPENAI_API_KEY=sk-your-api-key-here")
        print("    Linux:    export OPENAI_API_KEY=sk-your-api-key-here")
        print()
        sys.exit(1)

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)
    if not config and args.config.name != "config.json":
        print_error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not config:
        print_color("No config.json found, using defaults", Fore.YELLOW)
        print_color("Copy config.sample.json to config.json to customize", Fore.YELLOW)

    # Create and run app
    app = TestApp(
        config=config,
        api_key=api_key,
        text_only=args.text_only,
        debug=args.debug,
    )

    try:
        if args.script:
            if not args.script.exists():
                print_error(f"Script file not found: {args.script}")
                sys.exit(1)
            asyncio.run(app.run_script(args.script))
        else:
            asyncio.run(app.run_interactive())
    except KeyboardInterrupt:
        print("\n")
        print_info("Interrupted", "Goodbye!")


if __name__ == "__main__":
    main()

"""Script runner for automated testing."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a script step."""

    success: bool
    action: str
    description: str
    output: str = ""
    error: str | None = None


class ScriptRunner:
    """Runs sequential test scripts."""

    def __init__(
        self,
        tts_handler: Callable[[str], Any],
        stt_handler: Callable[[bytes | None, float | None], Any],
        chat_handler: Callable[[str], Any],
        record_handler: Callable[[float | None], Any],
        play_handler: Callable[[bytes, int], Any],
        load_audio_handler: Callable[[Path], tuple[bytes, int]],
    ) -> None:
        """Initialize script runner with handlers."""
        self.tts_handler = tts_handler
        self.stt_handler = stt_handler
        self.chat_handler = chat_handler
        self.record_handler = record_handler
        self.play_handler = play_handler
        self.load_audio_handler = load_audio_handler
        self._last_response: str = ""
        self._results: list[StepResult] = []

    def load_script(self, filepath: Path) -> dict:
        """Load a script from JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    async def run(self, script: dict) -> list[StepResult]:
        """Run a script and return results."""
        self._results = []
        name = script.get("name", "Unnamed Script")
        description = script.get("description", "")
        steps = script.get("steps", [])

        _LOGGER.info("Running script: %s", name)
        if description:
            _LOGGER.info("Description: %s", description)

        for i, step in enumerate(steps):
            _LOGGER.info("Step %d/%d: %s", i + 1, len(steps), step.get("action"))
            result = await self._execute_step(step)
            self._results.append(result)

            if not result.success:
                _LOGGER.error("Step failed: %s", result.error)
                if step.get("stop_on_error", True):
                    break

        return self._results

    async def _execute_step(self, step: dict) -> StepResult:
        """Execute a single step."""
        action = step.get("action", "unknown")
        description = step.get("description", action)

        try:
            if action == "log":
                message = step.get("message", "")
                _LOGGER.info("[SCRIPT] %s", message)
                return StepResult(True, action, description, output=message)

            elif action == "delay":
                seconds = step.get("seconds", 1)
                await asyncio.sleep(seconds)
                return StepResult(
                    True, action, description, output=f"Waited {seconds}s"
                )

            elif action == "tts":
                text = step.get("text", "")
                result = await self.tts_handler(text)
                if hasattr(result, "error") and result.error:
                    return StepResult(False, action, description, error=result.error)
                return StepResult(
                    True,
                    action,
                    description,
                    output=f"Generated TTS for: {text[:50]}...",
                )

            elif action == "stt":
                audio_file = step.get("audio_file")
                if audio_file:
                    audio_path = Path(audio_file)
                    if not audio_path.exists():
                        return StepResult(
                            False,
                            action,
                            description,
                            error=f"Audio file not found: {audio_file}",
                        )
                    audio_data, sample_rate = self.load_audio_handler(audio_path)
                    result = await self.stt_handler(audio_data, None)
                else:
                    duration = step.get("duration")
                    result = await self.stt_handler(None, duration)

                if hasattr(result, "error") and result.error:
                    return StepResult(False, action, description, error=result.error)

                transcript = getattr(result, "transcript", "") or getattr(
                    result, "text", ""
                )
                self._last_response = transcript
                return StepResult(
                    True, action, description, output=f"Transcription: {transcript}"
                )

            elif action == "record":
                duration = step.get("duration", 5)
                audio_data = await self.record_handler(duration)
                return StepResult(
                    True,
                    action,
                    description,
                    output=f"Recorded {len(audio_data)} bytes",
                )

            elif action == "type":
                text = step.get("text", "")
                result = await self.chat_handler(text)
                if hasattr(result, "error") and result.error:
                    return StepResult(False, action, description, error=result.error)
                response = getattr(result, "text", str(result))
                self._last_response = response
                return StepResult(
                    True, action, description, output=f"Response: {response[:100]}..."
                )

            elif action == "wait_response":
                timeout = step.get("timeout", 30)
                # In sequential mode, responses are already waited for
                return StepResult(True, action, description, output="Response received")

            elif action == "assert":
                contains = step.get("contains", "")
                not_contains = step.get("not_contains", "")

                if contains and contains.lower() not in self._last_response.lower():
                    return StepResult(
                        False,
                        action,
                        description,
                        error=f"Expected '{contains}' not found in response",
                    )
                if not_contains and not_contains.lower() in self._last_response.lower():
                    return StepResult(
                        False,
                        action,
                        description,
                        error=f"Unexpected '{not_contains}' found in response",
                    )
                return StepResult(True, action, description, output="Assertion passed")

            elif action == "play":
                audio_file = step.get("audio_file")
                if not audio_file:
                    return StepResult(
                        False, action, description, error="No audio file specified"
                    )
                audio_path = Path(audio_file)
                if not audio_path.exists():
                    return StepResult(
                        False,
                        action,
                        description,
                        error=f"Audio file not found: {audio_file}",
                    )
                audio_data, sample_rate = self.load_audio_handler(audio_path)
                await self.play_handler(audio_data, sample_rate)
                return StepResult(
                    True, action, description, output=f"Played {audio_file}"
                )

            else:
                return StepResult(
                    False, action, description, error=f"Unknown action: {action}"
                )

        except Exception as e:
            _LOGGER.exception("Step execution error")
            return StepResult(False, action, description, error=str(e))

    def get_summary(self) -> str:
        """Get a summary of script execution."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.success)
        failed = total - passed

        lines = [
            f"\n{'='*50}",
            f"Script Summary: {passed}/{total} steps passed",
            f"{'='*50}",
        ]

        for i, result in enumerate(self._results):
            status = "✓" if result.success else "✗"
            lines.append(
                f"  {status} Step {i+1}: {result.action} - {result.description}"
            )
            if result.error:
                lines.append(f"      Error: {result.error}")

        return "\n".join(lines)

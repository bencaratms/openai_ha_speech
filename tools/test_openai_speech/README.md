# OpenAI Speech Test Application

A standalone console application for testing STT (Speech-to-Text), TTS (Text-to-Speech), and Conversation functionality using OpenAI's APIs.

## Setup

1. Install dependencies:
   ```bash
   cd tools/test_openai_speech
   pip install -r requirements.txt
   ```

2. Set your API key (choose one method):
   
   **Option A**: Create a `.env` file:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```
   
   **Option B**: Set environment variable:
   ```bash
   # Windows
   set OPENAI_API_KEY=sk-your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY=sk-your-api-key-here
   ```

3. Copy config sample:
   ```bash
   cp config.sample.json config.json
   ```

## Usage

### Interactive Mode

```bash
# Default config
python main.py

# Custom config file
python main.py --config my_config.json

# Text-only mode (no audio devices)
python main.py --text-only

# Debug output
python main.py --debug
```

### Script Mode

```bash
python main.py --script test_script.json
```

## Interactive Commands

| Command | Description |
|---------|-------------|
| `tts <text>` | Convert text to speech |
| `stt` | Record and transcribe (Enter to stop) |
| `chat <text>` | Send text to conversation agent |
| `voice` | Voice conversation (speak → get response) |
| `mode <standard\|realtime>` | Switch API mode |
| `devices` | List audio devices |
| `config` | Show current configuration |
| `clear` | Clear chat history |
| `help` | Show available commands |
| `quit` | Exit application |

## Script Format

Scripts are JSON files with sequential steps:

```json
{
    "name": "Test Name",
    "description": "Test description",
    "steps": [
        {"action": "log", "message": "Starting..."},
        {"action": "tts", "text": "Hello world"},
        {"action": "delay", "seconds": 2},
        {"action": "type", "text": "What is 2+2?"},
        {"action": "wait_response", "timeout": 30},
        {"action": "assert", "contains": "4"}
    ]
}
```

### Available Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `log` | `message` | Print a log message |
| `delay` | `seconds` | Wait for specified time |
| `tts` | `text` | Generate and play TTS |
| `stt` | `audio_file` or `duration` | Transcribe audio |
| `record` | `duration` | Record audio for N seconds |
| `type` | `text` | Send chat message |
| `wait_response` | `timeout` | Wait for response |
| `assert` | `contains`, `not_contains` | Check response content |
| `play` | `audio_file` | Play an audio file |

## Output

All audio files are saved to the `output/` directory with timestamps:
- `recording_YYYYMMDD_HHMMSS.wav` - Microphone recordings
- `playback_YYYYMMDD_HHMMSS.wav` - Played audio
- `tts_realtime_YYYYMMDD_HHMMSS.wav` - Realtime TTS output
- `tts_standard_YYYYMMDD_HHMMSS.mp3` - Standard TTS output

## Notes

- Realtime API uses 24kHz audio; recordings are automatically resampled
- Standard TTS returns MP3/other formats; Realtime TTS returns PCM16
- The `voice` command only works in `realtime` mode
- Audio device issues? Try `--text-only` mode for testing chat/TTS text output

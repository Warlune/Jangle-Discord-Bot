# Jangle Discord Bot

Jangle is a local-first Discord voice and text AI bot. It can listen and speak in a Discord voice channel while serving text users at the same time, host party games, run restricted music controls, and connect to one of three local model backends:

- LM Studio, with no custom AI program required
- Ollama, with no custom AI program required
- Warlune, as an optional privacy-isolated guest adapter

Jangle does not require an OpenAI API key. Voice transcription runs locally with faster-whisper. The bot is intended to run on a computer you control, next to its local model server.

## Features

- Asynchronous voice and text conversations with isolated per-user temporary history
- Local Discord voice transcription through faster-whisper
- Fast Edge TTS output with multiple selectable voices
- Natural wake-word ownership, follow-up turns, and interruption handling
- Focused Game Mode with WoW trivia, general trivia, riddles, Would You Rather, and Twenty Questions
- Trivia buzz-ins while Jangle is still reading the question, without interrupting playback
- Voice polls, collaborative stories, server awards, and timed Party Mode
- YouTube song and playlist queues with administrator controls and DJ Mode
- Optional per-user notepads with strict size and sensitive-data filters
- Optional public web context from a local SearXNG server
- Test-mode JSONL logging for voice tuning, disabled by default

## Privacy Model

Jangle sends the configured model endpoint only:

- the current request;
- a short, in-memory conversation history for that Discord user;
- relevant notes that the same user explicitly saved in Jangle;
- public Discord context such as the current display name and voice-channel member names;
- optional public search snippets when SearXNG is enabled.

Jangle does not expose shell, filesystem, moderation, account, project, or browser tools to Discord users. Raw voice audio is not saved. Temporary conversation history disappears on restart or `/reset`.

When Warlune is selected, every Discord request uses guest mode with empty profile and memory context, no attachments, no action tools, and Warlune run logging disabled. Jangle-owned notes and settings remain in this project under `data/`; they never enter Warlune's profile or memory database.

The following local files are ignored by git:

- `.env` and `channels.env`;
- logs and JSONL conversation records;
- user notes, Discord IDs, voice choices, and personality state;
- downloaded model/tokenizer files and local databases;
- cookies, keys, and certificates.

Do not remove those ignore rules or commit local configuration files.

## Requirements

Required:

- Python 3.11
- A Discord bot application and token
- FFmpeg available on `PATH`
- One local model backend: LM Studio, Ollama, or Warlune
- A machine that can run faster-whisper and your chosen language model

Recommended:

- Deno on `PATH` for the most reliable current YouTube extraction through yt-dlp
- An NVIDIA GPU and working CUDA libraries for lower speech-recognition latency
- A test-only Discord text and voice channel while tuning

Python 3.12 or newer may work, but Python 3.11 is the tested target for the experimental Discord voice receive dependency.

## Quick Start

### Windows

```powershell
git clone https://github.com/Warlune/Jangle-Discord-Bot.git
cd Jangle-Discord-Bot
.\setup.ps1
notepad .env
.\start.ps1
```

`setup.ps1` creates `.venv`, installs [requirements.txt](requirements.txt), and copies `.env.example` to `.env` when needed.

### Linux or macOS

```bash
git clone https://github.com/Warlune/Jangle-Discord-Bot.git
cd Jangle-Discord-Bot
chmod +x setup.sh
./setup.sh
${EDITOR:-nano} .env
.venv/bin/python bot.py
```

Discord voice support still requires FFmpeg and platform audio dependencies. Linux package names vary by distribution.

## Discord Setup

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and create an application.
2. Add a bot and copy its token into `BOT_TOKEN` in `.env`.
3. Enable **Message Content Intent** on the bot page.
4. Invite the bot with the `bot` and `applications.commands` scopes.
5. Grant only View Channels, Send Messages, Read Message History, Connect, Speak, and Use Voice Activity.
6. Do not grant Administrator unless you have a separate reason unrelated to Jangle.
7. Create a test text channel and voice channel.
8. Enable Discord Developer Mode, copy the server and channel IDs, and place them in `.env`.

Recommended channel lock:

```dotenv
DISCORD_ALLOWED_GUILD_IDS=your_server_id
DISCORD_TEXT_CHANNEL_IDS=your_text_channel_id
DISCORD_TEXT_CHANNEL_NAMES=
DISCORD_VOICE_CHANNEL_IDS=your_voice_channel_id
DISCORD_VOICE_CHANNEL_NAMES=
```

Set `DISCORD_ADMIN_ROLE_IDS` when members with a particular role should be allowed to join/leave voice and operate administrator-only music controls. Server owners and members with Manage Server are also recognized as administrators.

Set `JANGLE_OWNER_USER_IDS` to immutable Discord user IDs allowed to use privacy controls such as ignoring a voice participant. This is separate from model provider ownership.

## Choose A Model Provider

### LM Studio

LM Studio is the easiest desktop option.

1. Install [LM Studio](https://lmstudio.ai/).
2. Download and load a chat/instruct model that fits your hardware.
3. Open the Developer tab and start the local server, or run `lms server start`.
4. Configure `.env`:

```dotenv
MODEL_PROVIDER=lmstudio
MODEL_ENDPOINT=http://127.0.0.1:1234/v1
MODEL_NAME=
MODEL_API_KEY=
```

When `MODEL_NAME` is blank, Jangle calls `/v1/models` and selects the first non-embedding model. Set it explicitly when several chat models are loaded. If LM Studio API authentication is enabled, place its local API token in `MODEL_API_KEY`.

See LM Studio's [local server guide](https://lmstudio.ai/docs/developer/core/server) and [OpenAI-compatible API guide](https://lmstudio.ai/docs/developer/openai-compat).

### Ollama

Ollama is a good CLI and headless option.

1. Install [Ollama](https://ollama.com/).
2. Pull a model and ensure Ollama is running:

```powershell
ollama pull llama3.2
ollama serve
```

3. Configure `.env`:

```dotenv
MODEL_PROVIDER=ollama
MODEL_ENDPOINT=http://127.0.0.1:11434/v1
MODEL_NAME=llama3.2
MODEL_API_KEY=
```

Ollama's local OpenAI-compatible endpoint accepts an API-key field but normally ignores it. Jangle communicates directly with `/v1/models` and `/v1/chat/completions`.

See Ollama's [OpenAI compatibility documentation](https://docs.ollama.com/api/openai-compatibility).

### Warlune

Warlune is optional. Public users do not need it.

```dotenv
MODEL_PROVIDER=warlune
WARLUNE_PATH=C:\path\to\warlune-lan-agent
WARLUNE_CONFIG=C:\path\to\warlune-lan-agent\config.json
WARLUNE_ENDPOINT=
WARLUNE_MODEL=
```

Blank endpoint/model values use Warlune's selected local model. Instant requests use its fast model path. `/medium` and `/pro` use its deeper orchestration while preserving the Discord guest privacy boundary.

## Optional Web Search

LM Studio and Ollama do not browse by themselves. Jangle can inject public search snippets from a SearXNG instance:

```dotenv
INTERNET_SEARCH_ENABLED=true
SEARXNG_URL=http://127.0.0.1:8888
SEARCH_MAX_RESULTS=5
```

Explicit search requests and freshness-sensitive prompts then use SearXNG's JSON endpoint. Search snippets are labeled as untrusted source material. Page contents are not fetched by the standalone adapter.

Warlune continues to use its own configured public-search route.

## Voice Configuration

The safe CPU defaults are:

```dotenv
WHISPER_MODEL=base.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=en
TTS_PROVIDER=edge
TTS_VOICE=en-US-AriaNeural
```

For NVIDIA CUDA, a common starting point is:

```dotenv
WHISPER_MODEL=small.en
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

If CUDA initialization fails, Jangle falls back to CPU `int8`.

Edge TTS sends only Jangle's generated answer text to Microsoft's speech service. Set `TTS_PROVIDER=none` to disable spoken output.

Pocket TTS is an advanced optional local voice path. Install its Python package with:

```powershell
.\setup.ps1 -PocketTts
```

Compatible Pocket model, tokenizer, and voice-state assets must be placed under `data/pocket-tts/`; these large files are deliberately not included in this repository. If Pocket is unavailable, its voice presets fall back to Edge TTS.

Useful tuning values:

```dotenv
VOICE_WAKE_WORDS=jangle,jengel,jingle,jangel
VOICE_SILENCE_MS=650
VOICE_MIN_MS=250
VOICE_MAX_SECONDS=25
VOICE_RMS_THRESHOLD=200
VOICE_PREROLL_MS=240
VOICE_BARGE_IN_FRAMES=15
VOICE_FOLLOWUP_SECONDS=25
```

`VOICE_RMS_THRESHOLD` controls microphone sensitivity; lower values hear quieter microphones but
may transcribe more background noise. `VOICE_PREROLL_MS` preserves the beginning of short wake-word
calls without delaying the response.

## Running Jangle

Start the bot, join the configured voice channel, then run `/join` in the configured text channel.

Windows:

```powershell
.\start.ps1
```

Any platform:

```bash
python bot.py
```

Jangle does not automatically reconnect to voice after a process restart. Run `/join` again.

## Text Commands

- `Jangle, <question>`, `@Jangle <question>`, or `/ask <question>` uses Instant.
- `/medium <question>` uses a larger answer budget and more careful instruction.
- `/pro <question>` uses the deepest configured route and largest answer budget.
- `/search <query>` forces configured public search when available.
- `/reset` clears only your temporary conversation history.
- `/remember`, `/memories`, and `/forget_memory` manage your private Jangle notepad.
- `/join` and `/leave` connect or disconnect voice.
- `/voice_debug` temporarily echoes recognized speech for tuning.
- `/activities` shows social commands.
- `/privacy` shows the runtime privacy boundary.
- `/status` shows provider, model, voice, logging, and request state.

Prefix fallbacks are `!ask`, `!medium`, `!pro`, and `!search` unless `BOT_PREFIX` is changed.

## Voice Games And Activities

Setup and control phrases require `Hey Jangle`. Participant responses are wake-free while their activity is active.

- `Hey Jangle, start WoW trivia`
- `Hey Jangle, start general trivia`
- `Hey Jangle, start riddles`
- `Hey Jangle, start Would You Rather`
- `Hey Jangle, start Twenty Questions`
- `Hey Jangle, hint`
- `Hey Jangle, game score`
- `Hey Jangle, next question`
- `Hey Jangle, stop game`
- `Hey Jangle, start poll raid or keys or battlegrounds`
- `Hey Jangle, start story mode about <theme>`
- `Hey Jangle, start an award for <category>`
- Administrator: `Hey Jangle, enable party mode for 15 minutes`
- Everyone: `Hey Jangle, party mode off`

Game Mode suppresses normal AI chat and unrelated controls until the game ends. Trivia players may answer while Jangle is reading; playback continues, and Discord audio onset determines the fastest correct answer. Each player gets one answer per attempt. If nobody is correct, Jangle repeats the same question once before revealing the answer and advancing.

## Personalities And Voices

Personality controls are administrator-only:

- `Hey Jangle, enable Savage mode`
- `Hey Jangle, enable Madam mode`
- `Hey Jangle, enable Brutal mode`
- `Hey Jangle, disable mode`

Savage is WoW-themed. Madam is authoritative, protective, diplomatic, and mature. Brutal enables adults-only profanity and aggressive consensual roasting while retaining safety boundaries.

Voice controls are administrator-only:

- `Hey Jangle, list voices`
- `Hey Jangle, change voice`
- `Hey Jangle, change voice to Brian`

## Music And DJ Mode

FFmpeg and yt-dlp handle YouTube playback. Live streams are excluded.

- Administrator: `Hey Jangle, enable DJ mode`
- Public while DJ Mode is active: natural requests containing a song and `queue`
- Public while DJ Mode is active: `Hey Jangle, show queue`
- Administrator: `Hey Jangle, play <song>`
- Administrator: `Hey Jangle, find playlist <name>`
- Administrator: `Hey Jangle, next`, `previous`, `pause`, or `resume`
- Administrator: `Hey Jangle, volume up`, `volume down`, or `set volume <0-100>`
- Administrator: `Hey Jangle, clear queue` or `stop`
- Administrator: `Hey Jangle, DJ mode off`

Music controls are locked while DJ Mode is off. DJ Mode ignores AI questions and accepts only
music controls; turning it off stops active music and clears the queue. Text chat remains available.

## Local Data And Test Logs

Jangle creates private runtime state under `data/`. These files may contain Discord server/user IDs and user-authored notes, so the whole directory is ignored by git.

Test logging is disabled by default. When enabled, Jangle records bot-directed text, accepted voice transcripts, answers, IDs/display names, and timing to a rotating local JSONL file. It does not save raw audio or ignored room conversation. Inform everyone in the channel before enabling it.

```dotenv
JANGLE_TEST_MODE=true
TEST_CONVERSATION_LOG=
TEST_LOG_MAX_MB=10
```

A blank log path uses `logs/test-conversations.jsonl`.

## Troubleshooting

**No model is available**

- LM Studio: load a chat model and start the Developer server.
- Ollama: run `ollama list`, pull a model, and set `MODEL_NAME` to its exact tag.
- Verify the configured `/v1/models` URL in a browser or with curl.

**Jangle is online but does not hear voice**

- Run `/join` while you are in an allowed voice channel.
- Check Connect, Speak, and Use Voice Activity permissions.
- Confirm FFmpeg is on `PATH`.
- Start with CPU Whisper settings before attempting CUDA.
- Use `/voice_debug true` briefly to inspect recognized text.

**Slash commands are slow to appear**

Set `DISCORD_DEV_GUILD_ID` to your test server while developing. Guild-scoped commands sync much faster than global commands.

**YouTube extraction fails**

Update yt-dlp, install current FFmpeg, and install Deno. YouTube changes frequently.

## Development

Install test dependencies and run the suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The tests use fake Discord/model objects and do not require a bot token, LM Studio, Ollama, or Warlune.

## License

Jangle is available under the [MIT License](LICENSE).

# speech-to-text

Microphone dictation → text, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
The same code runs on a PC with an NVIDIA GPU or on a laptop without one: it
**detects the hardware automatically** and picks the right model.

## Requirements

- [uv](https://docs.astral.sh/uv/) installed.
- A microphone.
- (Optional) An NVIDIA GPU with CUDA to run the large model at a good speed.

## Installation

```bash
git clone <REPO_URL>
cd speech-to-text
uv sync
```

`uv sync` creates the environment and installs everything (including the CUDA
libraries; on a machine without a GPU they are simply not used).

## Usage

```bash
uv run transcribe.py
```

- `[Enter]` → start recording 🔴
- Speak
- `[Enter]` → stop and **print the text** 📝
- `[q]` → quit

### Live mode

```bash
uv run transcribe.py --live
```

The text appears on a single line while you speak. The gray part is a draft
that keeps updating; every time you pause (~0.6 s) it becomes final (normal
color). Press the stop key to finish and get the full text (✅).

The stop key defaults to `Enter` and can be changed with `STT_STOP_KEY`
(`enter`, `space`, `esc`, `tab` or any single character):

```bash
# PowerShell
$env:STT_STOP_KEY="esc"; uv run transcribe.py --live

# bash
STT_STOP_KEY=esc uv run transcribe.py --live
```

The terminal must have focus for the stop key to be detected.

### Type mode (Windows)

```bash
uv run transcribe.py --type
```

Dictate into **any app** (editor, terminal, browser, chat box...). The script
runs in the background and types the text into the window that has focus:

1. Put the cursor where you want the text.
2. Press `Ctrl+Alt+Space` and speak. Each time you pause, that part is typed
   into the focused window (the live draft is shown in the script's console).
3. Finish with either:
   - `Enter`: types the last part and then presses `Enter` (e.g. sends the
     message).
   - `Ctrl+Alt+Space` again: types the last part without pressing anything.

While dictating, `Shift+Enter` and any other key reach the app normally. Press
`Ctrl+C` in the script's console to quit.

Configuration:

| Variable       | Default          | Example                          |
|----------------|------------------|----------------------------------|
| `STT_HOTKEY`   | `ctrl+alt+space` | `ctrl+shift+d`, `f9`, `win+alt+h` |
| `STT_STOP_KEY` | `enter`          | `esc`, `tab`                     |

Modifiers: `ctrl`, `alt`, `shift`, `win`. Keys: a letter/digit, `space`,
`enter`, `esc`, `tab`, `pause`, `insert`, `scrolllock`, `f1`–`f24`. Both keys
are captured system-wide while the script runs, so the focused app does not
receive them.

## Automatic model selection

| Detected hardware | Device | Default model |
|-------------------|--------|---------------|
| NVIDIA GPU (CUDA) | `cuda` | `large-v3`    |
| No GPU            | `cpu`  | `small`       |

### Forcing a model or device

Environment variables (useful e.g. to raise or lower quality):

```bash
# PowerShell
$env:STT_MODEL="medium"; uv run transcribe.py
$env:STT_DEVICE="cpu";   uv run transcribe.py

# bash
STT_MODEL=medium uv run transcribe.py
```

Available models (lightest to most accurate): `tiny`, `base`, `small`,
`medium`, `large-v3`.

### Language

Transcription defaults to **Spanish** (`es`). Change it with `STT_LANGUAGE`:

```bash
# PowerShell
$env:STT_LANGUAGE="en";   uv run transcribe.py   # English
$env:STT_LANGUAGE="auto"; uv run transcribe.py   # auto-detect on each recording

# bash
STT_LANGUAGE=en uv run transcribe.py
```

Any language code supported by Whisper works (`en`, `es`, `fr`, `de`, `pt`, ...).
Forcing a language is more reliable than `auto`, especially on short recordings.

On PowerShell, `$env:` variables persist for the rest of the terminal session;
use `Remove-Item Env:STT_LANGUAGE` to go back to the default.

## Tested on

| Machine | OS | CPU | RAM | GPU | Device / model |
|---------|----|-----|-----|-----|----------------|
| Desktop | Windows 11 Pro (build 26200) | Intel Core i5-12400F (6C/12T) | 32 GB | NVIDIA GeForce GTX 1070 8 GB (driver 582.66, CUDA 13.0) | `cuda` / `large-v3` (`int8`) |

Software: Python 3.13.3, faster-whisper 1.2.1, ctranslate2 4.8.1.

Languages: tested with **Spanish**, with very good results. **English has not
been tested yet.**

## Notes

- **Pascal GPUs (GTX 10xx)**: `compute_type=int8` is used (`int8_float16` is
  not supported on that architecture).
- **Windows + GPU**: the script automatically adds the cuBLAS/cuDNN DLLs
  installed by pip to `PATH` so ctranslate2 can find them.
- The first run downloads the model (it is cached for later runs).
- Whisper's typical hallucinations during silence are mitigated by the enabled
  VAD filter.

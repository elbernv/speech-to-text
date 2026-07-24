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

## Notes

- **Transcription language**: Spanish (`LANGUAGE = "es"` in `transcribe.py`).
- **Pascal GPUs (GTX 10xx)**: `compute_type=int8` is used (`int8_float16` is
  not supported on that architecture).
- **Windows + GPU**: the script automatically adds the cuBLAS/cuDNN DLLs
  installed by pip to `PATH` so ctranslate2 can find them.
- The first run downloads the model (it is cached for later runs).
- Whisper's typical hallucinations during silence are mitigated by the enabled
  VAD filter.

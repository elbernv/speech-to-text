"""
Microphone dictation -> text, using faster-whisper.

Usage:
    uv run transcribe.py

Flow:
    [Enter]  start recording
    [Enter]  stop recording and transcribe (prints the text)
    [q]      quit
"""

import os
import sys
import site

# --- Windows: make ctranslate2 find the CUDA DLLs (cuBLAS/cuDNN)
# installed by pip under site-packages/nvidia/**/bin.
# ctranslate2 loads these libraries lazily using the standard Windows
# search order, so the 'bin' folders must be added to PATH
# (add_dll_directory alone is not enough). This must run BEFORE importing
# faster_whisper / ctranslate2. ---
if sys.platform == "win32":
    _bin_dirs = []
    for _base in site.getsitepackages():
        _nvidia = os.path.join(_base, "nvidia")
        if os.path.isdir(_nvidia):
            for _root, _dirs, _files in os.walk(_nvidia):
                if os.path.basename(_root) == "bin":
                    _bin_dirs.append(_root)
                    try:
                        os.add_dll_directory(_root)
                    except OSError:
                        pass
    if _bin_dirs:
        os.environ["PATH"] = os.pathsep.join(_bin_dirs) + os.pathsep + os.environ.get("PATH", "")

import ctranslate2
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ----------------------------- Configuration -----------------------------
# Default model depends on the hardware. Can be forced with the STT_MODEL
# environment variable (e.g. STT_MODEL=medium) or STT_DEVICE (cuda / cpu).
# The language can be changed with STT_LANGUAGE (e.g. en, es, auto).
GPU_MODEL = "large-v3"         # with an NVIDIA GPU: best quality
CPU_MODEL = "small"            # no GPU (i5 laptop, etc.): fast and decent
DEFAULT_LANGUAGE = "es"        # forcing a language -> fewer language errors
SAMPLE_RATE = 16000            # Whisper's native format (16 kHz mono)


def get_language() -> str | None:
    """Transcription language from STT_LANGUAGE (e.g. 'en', 'es').

    'auto' lets Whisper detect the language on each recording (returns None).
    """
    language = os.environ.get("STT_LANGUAGE", "").strip().lower() or DEFAULT_LANGUAGE
    return None if language == "auto" else language


def detect_hardware() -> tuple[str, str]:
    """Pick the device and model based on what the machine has.

    - NVIDIA GPU (CUDA) available -> device='cuda', large model.
    - Otherwise -> device='cpu', light model.
    Both can be overridden with STT_DEVICE / STT_MODEL.
    """
    forced_device = os.environ.get("STT_DEVICE", "").strip().lower()
    forced_model = os.environ.get("STT_MODEL", "").strip()

    if forced_device in ("cuda", "cpu"):
        device = forced_device
    else:
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    model = forced_model or (GPU_MODEL if device == "cuda" else CPU_MODEL)
    return device, model


def load_model() -> WhisperModel:
    """Load the model on the best available device; fall back to CPU on failure."""
    device, model_size = detect_hardware()
    # 'int8' works on both CPU and Pascal GPUs (GTX 10xx). Do NOT use
    # 'int8_float16' on Pascal (not supported).
    try:
        print(f"Loading model '{model_size}' on {device.upper()}...")
        model = WhisperModel(model_size, device=device, compute_type="int8")
        print(f"Model loaded on {device.upper()}.\n")
        return model
    except Exception as e:
        if device == "cpu":
            raise
        print(f"Could not use the GPU ({e}).")
        print("Falling back to CPU (slower but works)...")
        fallback = os.environ.get("STT_MODEL", "").strip() or CPU_MODEL
        print(f"Loading model '{fallback}' on CPU...")
        model = WhisperModel(fallback, device="cpu", compute_type="int8")
        print("Model loaded on CPU.\n")
        return model


def record_until_enter() -> np.ndarray:
    """Record from the microphone until the user presses Enter."""
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input("  🔴 Recording... press [Enter] to stop ")

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()


def transcribe(model: WhisperModel, audio: np.ndarray) -> None:
    """Transcribe the audio with anti-hallucination settings and print it."""
    if audio.size == 0:
        print("  (no audio recorded)\n")
        return

    segments, _info = model.transcribe(
        audio,
        language=get_language(),
        vad_filter=True,                                   # trims silences
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,                  # don't carry over errors
        no_speech_threshold=0.6,
        temperature=0,                                     # deterministic
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    if text:
        print(f"\n  📝 {text}\n")
    else:
        print("  (no speech detected)\n")


def main() -> None:
    model = load_model()
    print(f"Language: {get_language() or 'auto-detect'}")
    print("=" * 60)
    print("  Dictation ready.  [Enter] record  |  [q] quit")
    print("=" * 60)

    while True:
        cmd = input("\n> [Enter] to record, [q] to quit: ")
        if cmd.strip().lower() == "q":
            print("Goodbye!")
            break
        audio = record_until_enter()
        transcribe(model, audio)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")

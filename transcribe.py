"""
Microphone dictation -> text, using faster-whisper.

Usage:
    uv run transcribe.py           # record, then transcribe
    uv run transcribe.py --live    # transcribe while you speak

Flow:
    [Enter]  start recording
    [Enter]  stop recording and transcribe (prints the text)
             (in --live mode the stop key is set with STT_STOP_KEY)
    [q]      quit
"""

import argparse
import os
import shutil
import site
import sys
import threading
import time

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
from faster_whisper.vad import VadOptions, get_speech_timestamps

if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty

# ----------------------------- Configuration -----------------------------
# Default model depends on the hardware. Can be forced with the STT_MODEL
# environment variable (e.g. STT_MODEL=medium) or STT_DEVICE (cuda / cpu).
# The language can be changed with STT_LANGUAGE (e.g. en, es, auto).
GPU_MODEL = "large-v3"         # with an NVIDIA GPU: best quality
CPU_MODEL = "small"            # no GPU (i5 laptop, etc.): fast and decent
DEFAULT_LANGUAGE = "es"        # forcing a language -> fewer language errors
SAMPLE_RATE = 16000            # Whisper's native format (16 kHz mono)

# Live mode (--live). The stop key is set with STT_STOP_KEY: enter (default),
# space, esc, tab or any single character.
LIVE_PAUSE_MS = 600            # pause that closes a segment and makes it final
LIVE_MAX_SEGMENT_S = 20        # force a cut if someone talks without pausing
LIVE_PARTIAL_INTERVAL_S = 0.5  # minimum time between partial updates
LIVE_VAD = VadOptions(min_silence_duration_ms=LIVE_PAUSE_MS, speech_pad_ms=200)
STOP_KEYS = {
    "enter": ("\r", "\n"),
    "space": (" ",),
    "esc": ("\x1b",),
    "tab": ("\t",),
}


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


def run_whisper(
    model: WhisperModel, audio: np.ndarray, beam_size: int = 5, prompt: str | None = None
) -> str:
    """Transcribe the audio with anti-hallucination settings and return the text."""
    segments, _info = model.transcribe(
        audio,
        language=get_language(),
        beam_size=beam_size,
        initial_prompt=prompt or None,                     # previous text as context
        vad_filter=True,                                   # trims silences
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,                  # don't carry over errors
        no_speech_threshold=0.6,
        temperature=0,                                     # deterministic
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(model: WhisperModel, audio: np.ndarray) -> None:
    """Transcribe the audio and print it."""
    if audio.size == 0:
        print("  (no audio recorded)\n")
        return

    text = run_whisper(model, audio)
    if text:
        print(f"\n  📝 {text}\n")
    else:
        print("  (no speech detected)\n")


# ------------------------------- Live mode -------------------------------
def parse_stop_key(name: str) -> tuple[str, ...]:
    """Map STT_STOP_KEY to the characters that key produces."""
    name = name.strip().lower() or "enter"
    if name in STOP_KEYS:
        return STOP_KEYS[name]
    if len(name) == 1:
        return (name,)
    raise SystemExit(
        f"Unsupported STT_STOP_KEY: {name!r}. Use one of {', '.join(STOP_KEYS)} "
        "or a single character."
    )


class KeyReader:
    """Non-blocking reads of single keypresses from the terminal."""

    def __enter__(self):
        if sys.platform != "win32":
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if sys.platform != "win32":
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self) -> str | None:
        """Return the next pressed key, or None if no key is waiting."""
        if sys.platform == "win32":
            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):                    # arrows, F-keys...
                msvcrt.getwch()
                return None
            return key
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def pressed(self, keys: tuple[str, ...]) -> bool:
        """Drain pending keypresses; True if any of them is one of `keys`."""
        hit = False
        while (key := self.read()) is not None:
            hit = hit or key.lower() in keys
        return hit


def show_live(final: str = "", draft: str = "") -> None:
    """Overwrite the current terminal line with the final text plus the
    (dimmed) draft. Long text is cut from the left so the end stays visible."""
    width = shutil.get_terminal_size().columns - 4
    draft = f" {draft}" if final and draft else draft
    if len(final) + len(draft) > width:
        keep = max(0, width - 1 - len(draft))
        final = "…" + final[-keep:] if keep else ""
        draft = draft[-(width - len(final)):]
    line = f"  {final}\x1b[2m{draft}\x1b[0m" if final or draft else ""
    sys.stdout.write("\r\x1b[2K" + line)
    sys.stdout.flush()


def find_cut(audio: np.ndarray) -> tuple[int, bool]:
    """Find how much of the buffer is finished.

    Returns (cut, has_speech): audio[:cut] is done and can be dropped (or
    made final if has_speech). cut == 0 means the speaker is still talking.
    """
    speech = get_speech_timestamps(audio, LIVE_VAD, sampling_rate=SAMPLE_RATE)
    if not speech:
        # Only silence: drop it, but keep a tail in case speech is starting.
        return max(0, audio.size - SAMPLE_RATE), False
    if speech[-1]["end"] < audio.size:
        # The VAD closed the last chunk, so there was a pause after it.
        return speech[-1]["end"], True
    if audio.size > LIVE_MAX_SEGMENT_S * SAMPLE_RATE:
        # No pause for too long: cut at the last gap between chunks, if any.
        if len(speech) > 1:
            return speech[-2]["end"], True
        return audio.size, True
    return 0, False


def live_session(model: WhisperModel, stop_keys: tuple[str, ...], key_label: str) -> str:
    """Transcribe while the user speaks until a stop key is pressed."""
    chunks: list[np.ndarray] = []
    lock = threading.Lock()
    buffer = np.zeros(0, dtype=np.float32)
    final: list[str] = []
    last_partial = 0.0

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with lock:
            chunks.append(indata[:, 0].copy())

    def drain() -> None:
        nonlocal buffer
        with lock:
            new = chunks[:]
            chunks.clear()
        if new:
            buffer = np.concatenate([buffer, *new])

    def commit(audio: np.ndarray) -> None:
        text = run_whisper(model, audio, prompt=" ".join(final)[-200:])
        if text:
            final.append(text)
        show_live(" ".join(final))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ), KeyReader() as keys:
        print(f"  🔴 Live... press [{key_label}] to stop")
        while not keys.pressed(stop_keys):
            drain()
            if buffer.size < SAMPLE_RATE // 2:
                time.sleep(0.05)
                continue

            cut, has_speech = find_cut(buffer)
            if cut:
                if has_speech:
                    commit(buffer[:cut])
                buffer = buffer[cut:]
            elif time.monotonic() - last_partial >= LIVE_PARTIAL_INTERVAL_S:
                draft = run_whisper(model, buffer, beam_size=1, prompt=" ".join(final)[-200:])
                show_live(" ".join(final), draft)
                last_partial = time.monotonic()
            else:
                time.sleep(0.05)

    drain()
    if buffer.size and get_speech_timestamps(buffer, LIVE_VAD, sampling_rate=SAMPLE_RATE):
        commit(buffer)
    show_live()
    return " ".join(final)


def main() -> None:
    parser = argparse.ArgumentParser(description="Microphone dictation -> text.")
    parser.add_argument(
        "--live", action="store_true", help="transcribe while you speak"
    )
    args = parser.parse_args()

    if args.live:
        key_label = (os.environ.get("STT_STOP_KEY", "").strip() or "enter").capitalize()
        stop_keys = parse_stop_key(key_label)
        if sys.platform == "win32":
            os.system("")                                  # enable ANSI escapes

    model = load_model()
    print(f"Language: {get_language() or 'auto-detect'}")
    print("=" * 60)
    if args.live:
        print(f"  Live dictation ready.  [{key_label}] stops  |  [q] quit")
    else:
        print("  Dictation ready.  [Enter] record  |  [q] quit")
    print("=" * 60)

    while True:
        cmd = input("\n> [Enter] to record, [q] to quit: ")
        if cmd.strip().lower() == "q":
            print("Goodbye!")
            break
        if args.live:
            text = live_session(model, stop_keys, key_label)
            print(f"  ✅ {text}\n" if text else "  (no speech detected)\n")
        else:
            audio = record_until_enter()
            transcribe(model, audio)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")

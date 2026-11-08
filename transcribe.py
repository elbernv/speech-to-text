"""
Microphone dictation -> text, using faster-whisper.

Usage:
    uv run transcribe.py           # record, then transcribe
    uv run transcribe.py --live    # transcribe while you speak
    uv run transcribe.py --type    # global hotkey, types into the focused window

Flow:
    [Enter]  start recording
    [Enter]  stop recording and transcribe (prints the text)
             (in --live mode the stop key is set with STT_STOP_KEY)
    [q]      quit
"""

import argparse
import ctypes
import os
import shutil
import site
import sys
import threading
import time
from collections.abc import Callable

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
DEFAULT_HOTKEY = "ctrl+alt+space"   # --type mode, change with STT_HOTKEY
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
        # The VAD closed the last chunk, so there was a pause after it. Keep
        # most of that silence (helps Whisper end the sentence with
        # punctuation), but not the newest bit: it may be the onset of the
        # next word that the VAD has not picked up yet.
        return max(speech[-1]["end"], audio.size - int(0.4 * SAMPLE_RATE)), True
    if audio.size > LIVE_MAX_SEGMENT_S * SAMPLE_RATE:
        # No pause for too long: cut at the last gap between chunks, if any.
        if len(speech) > 1:
            return speech[-2]["end"], True
        return audio.size, True
    return 0, False


def live_session(
    model: WhisperModel,
    should_stop: Callable[[], bool],
    on_final: Callable[[str], None] | None = None,
) -> str:
    """Transcribe while the user speaks until should_stop() returns True.

    on_final is called with each segment as soon as it becomes final.
    """
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
            if on_final:
                on_final(text)
        show_live(" ".join(final))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        while not should_stop():
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


# ------------------------------- Type mode -------------------------------
# Windows only: a global hotkey starts/stops live dictation and the final text
# is typed into whatever window has focus.
VK_NAMES = {
    "enter": 0x0D, "space": 0x20, "esc": 0x1B, "tab": 0x09, "pause": 0x13,
    "insert": 0x2D, "scrolllock": 0x91,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
}
MODIFIER_VKS = {"ctrl": (0x11,), "alt": (0x12,), "shift": (0x10,), "win": (0x5B, 0x5C)}
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
LLKHF_INJECTED = 0x10 | 0x02                               # sent by a program


def key_vk(name: str) -> int:
    """Windows virtual-key code for a key name (enter, f9, a, ...)."""
    name = name.strip().lower()
    if name in VK_NAMES:
        return VK_NAMES[name]
    if len(name) == 1:
        user32 = ctypes.WinDLL("user32")                   # own handle, own argtypes
        user32.VkKeyScanW.argtypes = [ctypes.c_wchar]
        user32.VkKeyScanW.restype = ctypes.c_short
        vk = user32.VkKeyScanW(name) & 0xFF
        if vk != 0xFF:
            return vk
    raise SystemExit(
        f"Unknown key: {name!r}. Use one of {', '.join(VK_NAMES)} or a single character."
    )


def parse_hotkey(spec: str) -> tuple[frozenset[str], int]:
    """'ctrl+alt+space' -> ({'ctrl', 'alt'}, VK_SPACE)."""
    *mods, key = [part.strip().lower() for part in spec.split("+")]
    unknown = set(mods) - MODIFIER_VKS.keys()
    if unknown:
        raise SystemExit(
            f"Unknown modifier(s) in STT_HOTKEY: {', '.join(sorted(unknown))}. "
            f"Use {', '.join(MODIFIER_VKS)}."
        )
    return frozenset(mods), key_vk(key)


def held_modifiers() -> frozenset[str]:
    state = ctypes.windll.user32.GetAsyncKeyState
    return frozenset(
        mod for mod, vks in MODIFIER_VKS.items() if any(state(vk) & 0x8000 for vk in vks)
    )


class GlobalKeys:
    """System-wide hotkey (start/stop) and stop key.

    Both are swallowed so the focused app never sees them. The stop key only
    counts while dictating and without modifiers (so Shift+Enter still works);
    it is sent again once the last text has been typed.
    """

    def __init__(self, hotkey: str, stop_key: str):
        from pynput import keyboard

        self._mods, self._hot_vk = parse_hotkey(hotkey)
        self.stop_vk = key_vk(stop_key)
        self._swallowed: set[int] = set()
        self.toggle = threading.Event()
        self.active = False
        self.stopped_with_key = False
        self._listener = keyboard.Listener(win32_event_filter=self._filter)

    def start(self) -> None:
        self._listener.start()

    def _filter(self, msg, data) -> bool:
        if data.flags & LLKHF_INJECTED:                    # our own typing
            return True
        vk = data.vkCode
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            mods = held_modifiers()
            if vk == self._hot_vk and mods == self._mods:
                self._swallow(vk)
            if self.active and vk == self.stop_vk and not mods:
                self._swallow(vk, stop_key=True)
        elif vk in self._swallowed:                        # matching key-up
            self._swallowed.discard(vk)
            self._listener.suppress_event()
        return True

    def _swallow(self, vk: int, stop_key: bool = False) -> None:
        if vk not in self._swallowed:                      # ignore key repeat
            self._swallowed.add(vk)
            self.stopped_with_key = stop_key
            self.toggle.set()
        self._listener.suppress_event()


def wait_modifiers_released(timeout: float = 2.0) -> None:
    """Typing while Ctrl/Alt are still held would trigger shortcuts."""
    end = time.monotonic() + timeout
    while held_modifiers() and time.monotonic() < end:
        time.sleep(0.02)


def type_mode(model: WhisperModel, keys: GlobalKeys, hotkey: str, stop_label: str) -> None:
    from pynput import keyboard

    typer = keyboard.Controller()
    keys.start()
    print("=" * 60)
    print(f"  Type mode ready.  [{hotkey}] start/stop in any window")
    print(f"  [{stop_label}] while dictating: finish and press {stop_label}")
    print("  Ctrl+C here to quit")
    print("=" * 60)

    while True:
        while not keys.toggle.wait(0.2):                   # stays Ctrl+C-able
            pass
        keys.toggle.clear()
        keys.active = True
        print("\n  🔴 Live... typing into the focused window")

        typed = []

        def type_final(text: str) -> None:
            wait_modifiers_released()
            typer.type((" " if typed else "") + text)
            typed.append(text)

        text = live_session(model, keys.toggle.is_set, type_final)
        keys.active = False
        if keys.stopped_with_key:
            wait_modifiers_released()
            typer.tap(keyboard.KeyCode.from_vk(keys.stop_vk))
        keys.toggle.clear()
        print(f"  ✅ {text}" if text else "  (no speech detected)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microphone dictation -> text.")
    parser.add_argument(
        "--live", action="store_true", help="transcribe while you speak"
    )
    parser.add_argument(
        "--type",
        action="store_true",
        help="global hotkey; type the text into the focused window (Windows only)",
    )
    args = parser.parse_args()

    stop_label = (os.environ.get("STT_STOP_KEY", "").strip() or "enter").capitalize()
    if args.type:
        if sys.platform != "win32":
            parser.error("--type is only supported on Windows for now")
        hotkey = os.environ.get("STT_HOTKEY", "").strip() or DEFAULT_HOTKEY
        keys = GlobalKeys(hotkey, stop_label)
    elif args.live:
        stop_keys = parse_stop_key(stop_label)
    if (args.live or args.type) and sys.platform == "win32":
        os.system("")                                      # enable ANSI escapes

    model = load_model()
    print(f"Language: {get_language() or 'auto-detect'}")
    if args.type:
        type_mode(model, keys, hotkey, stop_label)
        return

    print("=" * 60)
    if args.live:
        print(f"  Live dictation ready.  [{stop_label}] stops  |  [q] quit")
    else:
        print("  Dictation ready.  [Enter] record  |  [q] quit")
    print("=" * 60)

    while True:
        cmd = input("\n> [Enter] to record, [q] to quit: ")
        if cmd.strip().lower() == "q":
            print("Goodbye!")
            break
        if args.live:
            with KeyReader() as keys:
                print(f"  🔴 Live... press [{stop_label}] to stop")
                text = live_session(model, lambda: keys.pressed(stop_keys))
            print(f"  ✅ {text}\n" if text else "  (no speech detected)\n")
        else:
            audio = record_until_enter()
            transcribe(model, audio)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")

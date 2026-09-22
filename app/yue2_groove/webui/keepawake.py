"""Keep the (Shadow) PC awake while the music server runs — Windows only.

Two layers, both invisible to the user:
  1. SetThreadExecutionState: tells Windows "system + display required" (no sleep,
     no screen-off) for as long as this process lives.
  2. A tiny input jiggle every ~50 s: the mouse moves 1 px and straight back, plus a
     press of F15 (a key no keyboard has).  Cloud PCs such as Shadow watch for user
     input; this counts as input without moving anything you would notice.

Turn it off with YUE2_KEEP_AWAKE=0 (app/handy-zugang.txt or .env).
"""

from __future__ import annotations

import logging
import os
import sys
import threading

log = logging.getLogger("yue2_groove")

INTERVAL = float(os.environ.get("YUE2_KEEP_AWAKE_SECONDS", "50"))

_started = False


def _jiggle_loop(stop: threading.Event) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
        ]

    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    MOUSEEVENTF_MOVE, KEYEVENTF_KEYUP, VK_F15 = 0x0001, 0x0002, 0x7E

    def mouse(dx, dy):
        inp = INPUT(type=0)
        inp.u.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, 0)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def key(up):
        inp = INPUT(type=1)
        inp.u.ki = KEYBDINPUT(VK_F15, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    log.info("keep-awake: on (every %.0fs a 1-px mouse nudge + F15; YUE2_KEEP_AWAKE=0 turns it off)", INTERVAL)
    while not stop.is_set():
        try:
            kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            mouse(1, 0)
            mouse(-1, 0)
            key(False)
            key(True)
        except Exception as exc:  # noqa: BLE001 — never take the server down
            log.warning("keep-awake: %s", exc)
        stop.wait(INTERVAL)


def start() -> threading.Event | None:
    """Start the keep-awake thread once (Windows, unless YUE2_KEEP_AWAKE=0)."""
    global _started
    if _started or sys.platform != "win32":
        return None
    if os.environ.get("YUE2_KEEP_AWAKE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    _started = True
    stop = threading.Event()
    threading.Thread(target=_jiggle_loop, args=(stop,), daemon=True, name="keep-awake").start()
    return stop

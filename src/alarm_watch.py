"""Loud alarm on the laptop speaker when the user is needed. Works with no internet.

Sounds when Kaggle has been unreachable for 10 minutes, or when outputs/state/ALARM exists (written by me
when something needs the user). Stops by itself once the connection is back or the file is removed.
"""
from __future__ import annotations

import ctypes
import sys
import time
import winsound

import requests

from embed_common import Paths, Log

CHECK_S = 30
FAILS_TO_ALARM = 20
VK_VOLUME_UP = 0xAF
KEYUP = 0x0002


def volume_up():
    for _ in range(50):
        ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, KEYUP, 0)


def online() -> bool:
    try:
        requests.get("https://api.kaggle.com", timeout=15)
        return True
    except requests.RequestException:
        return False


def siren(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        winsound.Beep(2000, 350)
        winsound.Beep(1200, 350)


def main():
    if "--test" in sys.argv:
        siren(3)
        return
    paths = Paths(False)
    log = Log(paths, "alarm")
    flag = paths.state / "ALARM"
    fails, sounding = 0, False
    log(f"alarm watch started; sounds after {FAILS_TO_ALARM * CHECK_S // 60} min offline or if {flag.name} exists")
    while True:
        fails = 0 if online() else fails + 1
        need = fails >= FAILS_TO_ALARM or flag.exists()
        if need and not sounding:
            log(f"ALARM on: offline_checks={fails} flag={flag.exists()}")
            volume_up()
        elif sounding and not need:
            log("ALARM off")
        sounding = need
        if sounding:
            siren(CHECK_S)
        else:
            time.sleep(CHECK_S)


if __name__ == "__main__":
    main()

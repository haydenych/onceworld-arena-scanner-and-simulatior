"""Debug and lightweight perf logging utilities."""

import time

DEBUG = False


def set_debug(enabled):
    global DEBUG
    DEBUG = bool(enabled)


def is_debug():
    return DEBUG


def debug_print(message):
    if DEBUG:
        print(message)


def debug_perf(label, start_ts):
    if not DEBUG:
        return
    elapsed_ms = (time.time() - float(start_ts)) * 1000.0
    pretty_label = str(label).replace(":total", " total")
    print(f"[perf] {pretty_label}: {elapsed_ms:.2f} ms")


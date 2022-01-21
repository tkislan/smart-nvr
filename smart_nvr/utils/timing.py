import time


def get_current_time_millis() -> int:
    return int(time.time() * 1000.0)

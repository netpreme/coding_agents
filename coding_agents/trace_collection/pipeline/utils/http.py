from __future__ import annotations

import time
import urllib.error
import urllib.request


def check_server_initialized(url: str, timeout: float) -> bool:
    """Poll `url` until it returns 2xx or `timeout` seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.1)
    return False

import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path: str | Path, *, timeout_seconds: float = 30.0, poll_interval_seconds: float = 0.2):
    """Simple inter-process lock using fcntl.flock (macOS/Linux).

    This is used to serialize high-risk operations like Google Search UI automation.

    Raises TimeoutError if lock is not acquired within timeout.
    """

    import fcntl

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with open(lock_path, "w") as f:
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if (time.time() - start) >= timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
                time.sleep(poll_interval_seconds)
        try:
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

"""Signed demo sessions and a process-wide failed sign-in limit."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from collections import deque
from threading import Lock

SESSION_SECONDS = 7 * 24 * 60 * 60
COOKIE_NAME = "pp_session"
failed_logins = deque()
login_lock = Lock()


def signing_key():
    secret = os.environ.get("SESSION_SECRET", "")
    if secret:
        return secret.encode("utf-8")
    # Streamlit has only APP_PASSWORD. Forging offline needs a valid cookie first,
    # which needs the password; changing that password signs everyone out.
    return hashlib.sha256(b"paperpilot-session:" + os.environ.get("APP_PASSWORD", "").encode("utf-8")).digest()


def sign(session_id, expires):
    message = f"{session_id}.{expires}"
    signature = hmac.new(signing_key(), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{message}.{signature}"


def read_session(cookie) -> str | None:
    if not os.environ.get("APP_PASSWORD") or not isinstance(cookie, str):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}\.[0-9]{1,12}\.[0-9a-f]{64}", cookie):
        return None
    session_id, expires, signature = cookie.split(".")
    expected = sign(session_id, expires).rsplit(".", 1)[1]
    if not hmac.compare_digest(signature, expected) or int(expires) <= time.time():
        return None
    return session_id


def login_error(password) -> str | None:
    # Keep the limit check and failure recording together so concurrent attempts
    # cannot all pass the check before any of them records a failure.
    with login_lock:
        now = time.monotonic()
        while failed_logins and failed_logins[0] <= now - 60:
            failed_logins.popleft()
        if len(failed_logins) >= 10:
            return "limited"
        if not hmac.compare_digest(password.encode("utf-8"), os.environ.get("APP_PASSWORD", "").encode("utf-8")):
            failed_logins.append(now)
            return "incorrect"
    return None

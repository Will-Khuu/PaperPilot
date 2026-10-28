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


def owner_enabled():
    visitor = os.environ.get("APP_PASSWORD", "")
    return all(os.environ.get(name) and os.environ[name] != visitor
               for name in ("SESSION_SECRET", "OWNER_PASSWORD"))


def signing_key(role="visitor"):
    secret = os.environ.get("SESSION_SECRET", "")
    if role == "owner":
        if not owner_enabled():
            raise ValueError("Owner access is disabled")
        # A private key prevents visitor forgery; including the owner password
        # signs owners out when that password changes.
        return hmac.new(secret.encode("utf-8"),
                        b"owner:" + os.environ["OWNER_PASSWORD"].encode("utf-8"), hashlib.sha256).digest()
    if secret:
        # Preserve the existing visitor key so API and Streamlit cookies stay valid.
        return secret.encode("utf-8")
    # Keep the visitor fallback for Streamlit installs with only APP_PASSWORD.
    # Visitors know this key, so it must never authorize the owner role.
    return hashlib.sha256(b"paperpilot-session:" + os.environ.get("APP_PASSWORD", "").encode("utf-8")).digest()


def sign(session_id, role, expires):
    message = f"{session_id}.{role}.{expires}"
    signature = hmac.new(signing_key(role), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{message}.{signature}"


def read_session(cookie) -> tuple[str, str] | None:
    if not os.environ.get("APP_PASSWORD") or not isinstance(cookie, str):
        return None
    # Old three-part cookies have no role. Reject them so users sign in again.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}\.(visitor|owner)\.[0-9]{1,12}\.[0-9a-f]{64}", cookie):
        return None
    session_id, role, expires, signature = cookie.split(".")
    if role == "owner" and not owner_enabled():
        return None
    expected = sign(session_id, role, expires).rsplit(".", 1)[1]
    if not hmac.compare_digest(signature, expected) or int(expires) <= time.time():
        return None
    return session_id, role


def check_login(password):
    # Keep the limit check and failure recording together so concurrent attempts
    # cannot all pass the check before any of them records a failure.
    with login_lock:
        now = time.monotonic()
        while failed_logins and failed_logins[0] <= now - 60:
            failed_logins.popleft()
        if len(failed_logins) >= 10:
            return None, "limited"
        entered = password.encode("utf-8")
        owner_password = os.environ.get("OWNER_PASSWORD", "")
        # Always compare both secrets, even when the first one matches.
        owner_matches = hmac.compare_digest(entered, owner_password.encode("utf-8"))
        visitor_matches = hmac.compare_digest(entered, os.environ.get("APP_PASSWORD", "").encode("utf-8"))
        if owner_enabled() and owner_matches:
            return "owner", None
        if not visitor_matches:
            failed_logins.append(now)
            return None, "incorrect"
    return "visitor", None

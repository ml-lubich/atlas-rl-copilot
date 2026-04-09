"""MiniMax Text API (chatcompletion_v2) — no third-party SDK."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE = "https://api.minimax.io"
DEFAULT_MODEL = "MiniMax-M2.5"
CHAT_PATH = "/v1/text/chatcompletion_v2"


def minimax_chat(
    *,
    system_text: str,
    user_text: str,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> str:
    """
    Call MiniMax chat completion v2. Requires MINIMAX_API_KEY unless api_key is passed.

    Docs: POST /v1/text/chatcompletion_v2 (Bearer token).
    """
    key = api_key or os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MINIMAX_API_KEY is not set")

    base = (api_base or os.environ.get("MINIMAX_API_BASE") or DEFAULT_BASE).rstrip("/")
    mdl = model or os.environ.get("MINIMAX_MODEL") or DEFAULT_MODEL

    url = f"{base}{CHAT_PATH}"
    payload: dict[str, Any] = {
        "model": mdl,
        "messages": [
            {"role": "system", "name": "Atlas", "content": system_text},
            {"role": "user", "name": "user", "content": user_text},
        ],
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax HTTP {exc.code}: {detail}") from exc

    data = json.loads(raw)
    base_resp = data.get("base_resp") or {}
    code = base_resp.get("status_code", 0)
    if code != 0:
        msg = base_resp.get("status_msg", "unknown error")
        raise RuntimeError(f"MiniMax API error status_code={code}: {msg}")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("MiniMax response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError("MiniMax response missing message.content")
    return str(content)

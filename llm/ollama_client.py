"""HTTP client to local Ollama (streaming optional)."""

import json
import logging

import requests

logger = logging.getLogger(__name__)


def _parse_tool_calls(raw: list) -> list[dict]:
    """Normalize tool_calls from Ollama response: each has 'name' and 'arguments' (dict)."""
    out = []
    for tc in raw or []:
        fn = tc if isinstance(tc, dict) else getattr(tc, "__dict__", {})
        f = fn.get("function") or {}
        name = f.get("name", "") if isinstance(f, dict) else ""
        args = f.get("arguments") if isinstance(f, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        out.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return out


def _is_oom_error(response) -> bool:
    """True if response body indicates GPU OOM / allocation failure."""
    try:
        text = (response.text or "").lower()
        return (
            "allocate" in text
            or "buffer" in text
            or "failed to load model" in text
            or "out of memory" in text
        )
    except Exception:
        return False


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    stream: bool = False,
    num_ctx: int = 1024,
) -> str:
    """Send chat request to Ollama; return full response content.
    num_ctx limits context size to reduce GPU memory (KV cache) on 8GB Jetson.
    On 500 with OOM, retries once with num_ctx=512."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_ctx": num_ctx},
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code == 200:
            data = r.json()
            return (data.get("message") or {}).get("content", "")
        if r.status_code == 500 and _is_oom_error(r) and num_ctx > 512:
            logger.warning(
                "Ollama GPU OOM (num_ctx=%s). Retrying with num_ctx=512.",
                num_ctx,
            )
            payload["options"]["num_ctx"] = 512
            r2 = requests.post(url, json=payload, timeout=120)
            if r2.status_code == 200:
                data = r2.json()
                return (data.get("message") or {}).get("content", "")
            logger.warning(
                "Ollama still failed after reducing context. Free GPU: sudo scripts/prepare-ollama-gpu.sh; restart ollama; or use smaller model (e.g. OLLAMA_MODEL=llama3.2:1b)."
            )
            return ""
        r.raise_for_status()
        return ""
    except requests.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None and _is_oom_error(e.response):
            logger.warning(
                "Ollama GPU OOM. Free memory: sudo scripts/prepare-ollama-gpu.sh; then restart ollama. Or use smaller model."
            )
        else:
            logger.warning("Ollama request failed: %s", err_msg)
        return ""


def chat_with_tools(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    stream: bool = False,
    num_ctx: int = 1024,
) -> dict:
    """Send chat request with tools; return dict with 'content' (str) and 'tool_calls' (list of {name, arguments})."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "tools": tools,
        "options": {"num_ctx": num_ctx},
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code != 200:
            if r.status_code == 500 and _is_oom_error(r) and num_ctx > 512:
                payload["options"]["num_ctx"] = 512
                r2 = requests.post(url, json=payload, timeout=120)
                if r2.status_code == 200:
                    data = r2.json()
                    msg = data.get("message") or {}
                    return {
                        "content": msg.get("content", ""),
                        "tool_calls": _parse_tool_calls(msg.get("tool_calls") or []),
                    }
            return {"content": "", "tool_calls": []}
        data = r.json()
        msg = data.get("message") or {}
        return {
            "content": (msg.get("content") or "").strip(),
            "tool_calls": _parse_tool_calls(msg.get("tool_calls") or []),
        }
    except requests.RequestException as e:
        logger.warning("Ollama chat_with_tools failed: %s", e)
        return {"content": "", "tool_calls": []}


def is_ollama_available(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Check if Ollama API is reachable."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def is_ollama_model_available(base_url: str, model: str) -> bool:
    """Check if Ollama is reachable and the given model is pulled (avoids 500 on chat)."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        data = r.json()
        models = data.get("models") or []
        for m in models:
            name = m.get("name") or ""
            if name == model:
                return True
        return False
    except Exception:
        return False

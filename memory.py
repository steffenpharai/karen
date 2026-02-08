"""Long-term memory: rolling summary and session persistence for the orchestrator."""

import json
import logging
from datetime import datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

SUMMARY_FILE = "session_summary.json"


def _summary_path(base_dir: Path) -> Path:
    """Path to session summary JSON file."""
    return Path(base_dir) / SUMMARY_FILE


def load_summary(base_dir: Path) -> str:
    """Load rolling summary from JSON; return empty string if missing or invalid."""
    path = _summary_path(base_dir)
    if not path.exists():
        return ""
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("summary", "") or ""
    except Exception as e:
        logger.warning("Load summary failed: %s", e)
        return ""


def save_summary(base_dir: Path, summary: str) -> None:
    """Write rolling summary to JSON. Creates base_dir if needed."""
    path = _summary_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = {"summary": summary, "updated_at": datetime.now().isoformat()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("Save summary failed: %s", e)


def load_session(data_dir: str | Path) -> dict:
    """Load session state: summary and reminders path. Returns dict with 'summary' and 'data_dir'."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(data_dir)
    return {"summary": summary, "data_dir": data_dir}


def save_session(memory: dict) -> None:
    """Persist session: write rolling summary to disk."""
    data_dir = memory.get("data_dir")
    if not data_dir:
        return
    summary = memory.get("summary", "")
    save_summary(Path(data_dir), summary)


def maybe_summarize(
    memory: dict,
    short_term_turns: list[dict],
    base_url: str,
    model: str,
    num_ctx: int = 1024,
    every_n_turns: int | None = None,
) -> None:
    """If short_term has enough turns, ask LLM to summarize and update rolling summary."""
    every_n = every_n_turns or settings.SUMMARY_EVERY_N_TURNS
    if len(short_term_turns) < every_n:
        return
    from llm.ollama_client import chat

    # Build a compact transcript of last N turns for summarization
    lines = []
    for msg in short_term_turns[-every_n:]:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Jarvis"
        lines.append(f"{prefix}: {content[:200]}")
    if not lines:
        return
    transcript = "\n".join(lines)
    prompt = (
        "Summarize this brief conversation in one or two short sentences, "
        "preserving key facts, requests, and outcomes. Output only the summary, no preamble.\n\n"
        f"{transcript}"
    )
    messages = [
        {"role": "system", "content": "You are a concise summarizer. Output only the summary text."},
        {"role": "user", "content": prompt},
    ]
    try:
        new_summary = chat(base_url, model, messages, stream=False, num_ctx=num_ctx)
        if new_summary and new_summary.strip():
            prev = memory.get("summary", "")
            memory["summary"] = (prev + " " + new_summary.strip()).strip() if prev else new_summary.strip()
            memory["summary"] = memory["summary"][:1500]  # cap length
            save_session(memory)
            logger.debug("Updated rolling summary (%s chars)", len(memory["summary"]))
    except Exception as e:
        logger.warning("Summarization failed: %s", e)

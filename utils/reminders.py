"""Local file-based reminders."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_reminders_path(base_dir: Path) -> Path:
    """Path to reminders JSON file."""
    return base_dir / "reminders.json"


def load_reminders(base_dir: Path) -> list[dict]:
    """Load reminders from JSON; return list of {text, done, ...}."""
    path = get_reminders_path(base_dir)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Load reminders failed: %s", e)
        return []


def format_reminders_for_llm(reminders: list[dict], max_items: int = 5) -> str:
    """Format pending reminders for LLM context."""
    pending = [r for r in reminders if not r.get("done")][:max_items]
    return "; ".join(r.get("text", "") for r in pending) if pending else ""

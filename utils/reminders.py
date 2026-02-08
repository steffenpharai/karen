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


def save_reminders(base_dir: Path, reminders: list[dict]) -> None:
    """Write reminders to JSON file. Creates base_dir if needed."""
    path = get_reminders_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(reminders, f, indent=2)
    except Exception as e:
        logger.warning("Save reminders failed: %s", e)


def add_reminder(base_dir: Path, text: str, time_str: str = "") -> None:
    """Append one reminder and save. time_str is optional (e.g. '14:00' or 'tomorrow')."""
    base_dir = Path(base_dir)
    reminders = load_reminders(base_dir)
    reminders.append({"text": text, "time": time_str, "done": False})
    save_reminders(base_dir, reminders)

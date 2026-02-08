"""nvpmodel -q, jetson_clocks, thermal check for Jetson Orin."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_power_mode() -> str | None:
    """Return current nvpmodel mode (e.g. MAXN_SUPER)."""
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug("nvpmodel -q failed: %s", e)
        return None


def get_tegrastats_sample() -> str | None:
    """One-line tegrastats sample (if available)."""
    try:
        out = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def get_thermal_warning() -> str | None:
    """Return a warning message if thermal throttling likely (e.g. from tegrastats)."""
    sample = get_tegrastats_sample()
    if not sample:
        return None
    # Simple heuristic: check for "thermal" or "throttle" in output
    lower = sample.lower()
    if "throttle" in lower or "thermal" in lower:
        return "Thermal/throttling indicated; consider cooling or reducing load."
    return None


def get_system_stats() -> str | None:
    """Return a one-line system stats string for LLM context (power mode + tegrastats/jtop)."""
    parts = []
    pm = get_power_mode()
    if pm:
        parts.append(f"Power: {pm}")
    try:
        out = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout:
            # First line often has RAM, GPU, CPU
            line = out.stdout.strip().split("\n")[0].strip()
            if line:
                parts.append(f"Stats: {line[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return "; ".join(parts) if parts else None

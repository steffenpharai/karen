"""Simple status overlay with optional vision preview (Tkinter)."""

import logging
import os
import threading
from tkinter import Label, Tk
from tkinter import font as tkfont

logger = logging.getLogger(__name__)

# Shared state for status and latest frame path (optional)
_status = "Idle"
_status_lock = threading.Lock()
_latest_frame_path: str | None = None


def set_status(value: str) -> None:
    with _status_lock:
        global _status
        _status = value


def get_status() -> str:
    with _status_lock:
        return _status


def set_latest_frame_path(path: str | None) -> None:
    with _status_lock:
        global _latest_frame_path
        _latest_frame_path = path


def get_latest_frame_path() -> str | None:
    with _status_lock:
        return _latest_frame_path


def run_overlay(update_interval_ms: int = 200) -> None:
    """Run the status overlay window (blocking). Call from main thread or a dedicated GUI thread."""
    root = Tk()
    root.title("Jarvis")
    root.geometry("320x280")
    root.configure(bg="black")
    root.attributes("-topmost", True)

    label = Label(
        root,
        text="Idle",
        fg="lime",
        bg="black",
        font=tkfont.Font(family="Monospace", size=14),
    )
    label.pack(expand=False, pady=4)

    img_label = Label(root, bg="black")
    img_label.pack(expand=True)

    def refresh():
        label.config(text=get_status())
        path = get_latest_frame_path()
        if path and os.path.isfile(path):
            try:
                from PIL import Image, ImageTk

                img = Image.open(path)
                img.thumbnail((320, 180))
                ph = ImageTk.PhotoImage(img)
                img_label.config(image=ph)
                img_label.image = ph
            except ImportError:
                pass
            except Exception:
                pass
        root.after(update_interval_ms, refresh)

    root.after(update_interval_ms, refresh)
    try:
        root.mainloop()
    except Exception as e:
        logger.warning("Overlay closed: %s", e)

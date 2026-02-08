"""Build prompt with vision + reminders for Jarvis."""


def build_messages(
    system_prompt: str,
    user_text: str,
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
) -> list[dict]:
    """Build chat messages for Ollama: system + optional vision/reminders/time/stats + user."""
    parts = []
    if current_time:
        parts.append(f"[Current time: {current_time}]")
    if system_stats:
        parts.append(f"[System: {system_stats}]")
    if vision_description:
        parts.append(f"[Scene: {vision_description}]")
    if reminders_text:
        parts.append(f"[Reminders: {reminders_text}]")
    content = user_text
    if parts:
        content = "\n\n".join(parts) + "\n\n" + content
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

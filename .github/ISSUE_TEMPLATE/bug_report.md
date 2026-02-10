---
name: Bug Report
about: Something isn't working as expected
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

A clear description of what the bug is.

## Environment

- **Jetson model**: Orin Nano Super 8GB / other
- **JetPack version**: 6.x (run `cat /etc/nv_tegra_release`)
- **Power mode**: MAXN_SUPER / other (run `sudo nvpmodel -q`)
- **Python version**: (run `python3 --version`)
- **Ollama model**: qwen3:1.7b / other
- **Run mode**: `--serve` / `--orchestrator` / `--e2e` / other

## Steps to Reproduce

1. ...
2. ...
3. ...

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include error messages or logs.

## Logs

<details>
<summary>Relevant logs</summary>

```
Paste logs here
```

</details>

## Additional Context

- RAM usage (`free -h` output)
- GPU status (`ollama ps` output)
- Thermal state (`cat /sys/devices/virtual/thermal/thermal_zone*/temp`)

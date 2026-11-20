"""Environment-backed application configuration.

Environment variables take precedence over values in the repository-root
``.env`` file. The file is intended for local development only and is never
required in CI or production deployments that provide environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE entries without overwriting existing variables."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def get_env(name: str, default: str | os.PathLike[str] = "") -> str:
    """Return a trimmed environment value after local configuration loading."""
    value = os.environ.get(name)
    return value.strip() if value is not None else str(default).strip()


load_env_file()

"""Create offline backups of the football and NBA SQLite databases.

Example:
    python v4_web/scripts/backup_databases.py --output-dir backups
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from database_maintenance import backup_database


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "v4_web"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory outside source control for backup files",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    databases = {
        "football": WEB_ROOT / "data" / "predictions.db",
        "nba": WEB_ROOT / "predictions.db",
    }
    for name, source in databases.items():
        destination = args.output_dir / f"{name}-predictions-{timestamp}.db"
        backup_database(source, destination)
        print(f"Backed up {name}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

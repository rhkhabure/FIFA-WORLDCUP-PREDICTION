# NBA play-by-play database

`nba_stats_pbp.sqlite` is intentionally kept out of Git because it is a
multi-gigabyte dataset. Download it from the Kaggle dataset
[NBA Stats Dataset: 1996 - Present](https://www.kaggle.com/datasets/chevronronson/nba-stats-dataset),
which is listed as CC0, and place the file in this directory.

The audited source was Kaggle dataset version 57 (updated 2026-08-01). Its
`sql/nba_stats_pbp.sqlite` file is listed as 5,574,262,784 bytes. The database
contains the `play_by_play` table; the audited copy contains 18,256,950 rows.
Kaggle's file listing does not provide a checksum, so validate the downloaded
file by its size, table, row count, and SQLite quick check before using it.

To download the audited version with KaggleHub:

```python
from pathlib import Path
import kagglehub
import shutil
import sqlite3

download_dir = Path(
    kagglehub.dataset_download(
        "chevronronson/nba-stats-dataset/versions/57"
    )
)
source = download_dir / "sql" / "nba_stats_pbp.sqlite"
target = Path("v4_web/nba/sql/nba_stats_pbp.sqlite")

if source.stat().st_size != 5_574_262_784:
    raise ValueError(f"Unexpected database size: {source.stat().st_size}")

shutil.copy2(source, target)
connection = sqlite3.connect(
    f"file:{target.resolve().as_posix()}?mode=ro&immutable=1",
    uri=True,
)
try:
    if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise ValueError("SQLite quick_check failed")
    if connection.execute(
        "SELECT COUNT(*) FROM play_by_play"
    ).fetchone()[0] != 18_256_950:
        raise ValueError("Unexpected play_by_play row count")
finally:
    connection.close()
```

The Git-tracked version did not match this full-size dataset and must not be
used as a substitute. Historical shot analysis must also account for
incomplete 2025-26 play-by-play coverage; use the NBA API for the 2025-26 and
2026-27 seasons.

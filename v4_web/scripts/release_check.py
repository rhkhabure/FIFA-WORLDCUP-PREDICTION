"""Verify public deployment endpoints after a release."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


CHECKS = {
    "/health/live": '"status":"alive"',
    "/health/ready": '"status":"ready"',
    "/privacy": "Privacy notice",
    "/robots.txt": "Sitemap:",
    "/sitemap.xml": "<urlset",
}


def fetch(base_url: str, path: str, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"User-Agent": "melios-release-check/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Public HTTPS origin")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    failed = False
    for path, marker in CHECKS.items():
        try:
            status, body = fetch(args.base_url, path, args.timeout)
            passed = status == 200 and marker in body
            print(f"{'PASS' if passed else 'FAIL'} {path} status={status}")
        except (OSError, urllib.error.URLError, UnicodeError) as error:
            failed = True
            print(f"FAIL {path}: {error}")
            continue
        failed = failed or not passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

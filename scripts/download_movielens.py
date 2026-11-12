"""Download MovieLens ml-latest-small (100k ratings, ~9.7k movies) into data/.

    python scripts/download_movielens.py

MovieLens is licensed by GroupLens for research use and may not be redistributed,
so this repo never commits the data. If the automatic download fails (for
example on a TLS error, which some networks hit with files.grouplens.org),
download the zip in a browser from
https://grouplens.org/datasets/movielens/latest/ and pass it with --zip.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, help="Use an already-downloaded ml-latest-small.zip")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    if args.zip:
        payload = args.zip.read_bytes()
    else:
        print(f"Downloading {URL} ...")
        try:
            with urllib.request.urlopen(URL, timeout=60) as resp:
                payload = resp.read()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Download failed ({exc}).\nDownload the zip in a browser from "
                "https://grouplens.org/datasets/movielens/latest/ then run:\n"
                "  python scripts/download_movielens.py --zip path/to/ml-latest-small.zip"
            ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(args.out)
    print(f"Extracted to {args.out / 'ml-latest-small'}")


if __name__ == "__main__":
    main()

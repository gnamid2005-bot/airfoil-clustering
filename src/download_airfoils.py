import argparse
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve


DEFAULT_URL = "https://m-selig.web.engr.illinois.edu/ads/archives/coord_seligFmt.zip"
DEFAULT_RAW_DIR = Path("data/airfoils/raw")
ARCHIVE_NAME = "coord_seligFmt.zip"


def safe_extract_zip(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        archive.extractall(extract_dir)


def download_archive(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        print(f"archive already exists: {archive_path}")
        return

    try:
        print(f"downloading {url}")
        urlretrieve(url, archive_path)
    except URLError as exc:
        raise RuntimeError(
            "Failed to download the UIUC airfoil archive. "
            f"Place {ARCHIVE_NAME} manually in {archive_path.parent} and rerun."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and safely extract the UIUC airfoil coordinate archive."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    archive_path = raw_dir / ARCHIVE_NAME
    extract_dir = raw_dir / "coord_seligFmt"

    download_archive(args.url, archive_path)

    existing = list(extract_dir.rglob("*.dat")) if extract_dir.exists() else []
    if existing and not args.force_extract:
        print(f"extracted files already exist: {extract_dir}")
        return

    safe_extract_zip(archive_path, extract_dir)
    dat_count = len(list(extract_dir.rglob("*.dat")))
    print(f"extracted {dat_count} .dat files to {extract_dir}")


if __name__ == "__main__":
    main()

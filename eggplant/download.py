"""Verified download of the Mendeley eggplant dataset (original images only)."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from .utils import log

API_DATASET = "https://data.mendeley.com/public-api/datasets/{id}?version={ver}"
API_FILES = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={ver}"


UA = {"User-Agent": "eggplant-disease-supcon-xai/1.0 (research download script)"}


def _transient(e: Exception) -> bool:
    return not (isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429)


def _get_json(url: str, retries: int = 3) -> object:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries or not _transient(e):
                raise
            log(f"network error ({e}); retry {attempt}/{retries - 1}")
            time.sleep(5 * attempt)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, size: int, retries: int = 3) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers=UA | ({"Range": f"bytes={have}-"} if have else {}))
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored Range: restart
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as f, tqdm(total=size, initial=have, unit="B", unit_scale=True,
                                                  desc=f"download {dest.name}") as bar:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        f.write(block)
                        bar.update(len(block))
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries or not _transient(e):
                raise
            log(f"download interrupted ({e}); resume attempt {attempt}/{retries - 1}")
            time.sleep(5 * attempt)


def acquire(cfg: dict, local_zip: str | None = None) -> Path:
    """Download + verify + extract the original-image archive. Returns extraction dir."""
    d = cfg["data"]
    raw = Path(d["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    meta = _get_json(API_DATASET.format(id=d["mendeley_id"], ver=d["version"]))
    files = _get_json(API_FILES.format(id=d["mendeley_id"], ver=d["version"]))
    source = {
        "name": meta["name"], "doi": meta["doi"]["id"], "version": meta["version"],
        "publish_date": meta.get("publish_date"), "license": meta["data_licence"]["short_name"],
        "license_url": meta["data_licence"]["url"],
        "files": [{"filename": f["filename"], "size": f["size"], "sha256": f["content_details"]["sha256_hash"]}
                  for f in files],
    }
    (raw / "source_metadata.json").write_text(json.dumps(source, indent=2))
    log(f"source: {source['name']} | DOI {source['doi']} | v{source['version']} | {source['license']}")
    entry = next(f for f in files if f["filename"] == d["archive_name"])
    zpath = raw / d["archive_name"]
    expected = entry["content_details"]["sha256_hash"]
    if local_zip:
        zpath = Path(local_zip)
        if sha256_file(zpath) != expected:
            raise RuntimeError(f"{zpath} does not match the published SHA-256 {expected}")
        log(f"local archive verified: {zpath}")
    elif zpath.exists() and sha256_file(zpath) == expected:
        log(f"cached archive verified: {zpath}")
    else:
        _download(entry["content_details"]["download_url"], zpath, entry["size"])
        log("verifying SHA-256 ...")
        got = sha256_file(zpath)
        if got != expected:
            raise RuntimeError(f"SHA-256 mismatch for {zpath}: got {got}, expected {expected}")
        log("SHA-256 verified")
    out = raw / "original"
    if not (out / ".extracted").exists():
        with zipfile.ZipFile(zpath) as z:
            for m in tqdm(z.infolist(), desc="extract", unit="file"):
                if ".." in Path(m.filename).parts or m.filename.startswith("/"):
                    raise RuntimeError(f"unsafe zip member {m.filename}")
                z.extract(m, out)
        (out / ".extracted").write_text(expected)
    log(f"extracted to {out}")
    return out

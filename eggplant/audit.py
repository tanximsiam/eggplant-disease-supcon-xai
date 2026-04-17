"""Image manifest + integrity / duplicate audit (runs once, before any split)."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from .utils import atomic_json, file_sha256, log

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VARIANTS = ["id", "hflip", "vflip", "rot90", "rot180", "rot270"]
EXPECTED_TOTAL = 3116  # stated by the source; compared, never assumed


def find_image_root(raw_original: Path) -> Path:
    """Directory whose child folders are the class folders."""
    cands = [p for p in raw_original.rglob("*") if p.is_dir()
             and any(c.is_file() and c.suffix.lower() in IMG_EXT for c in p.iterdir())]
    if not cands:
        raise RuntimeError(f"no image folders under {raw_original}")
    parents = {p.parent for p in cands}
    if len(parents) != 1:
        raise RuntimeError(f"ambiguous class-folder layout: {sorted(map(str, parents))}")
    return parents.pop()


def _variants(img: Image.Image) -> list[Image.Image]:
    return [img, ImageOps.mirror(img), ImageOps.flip(img), img.rotate(90, expand=True),
            img.rotate(180), img.rotate(270, expand=True)]


def _process(args) -> dict:
    path, root, cache_dir, short_side = args
    rec = {"rel_path": str(Path(path).relative_to(root)), "label_original": Path(path).parent.name}
    rec["file_bytes"] = os.path.getsize(path)
    rec["sha256"] = file_sha256(path)
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rec["format"] = im.format
            rec["mode_original"] = im.mode
            rec["exif_orientation"] = int(im.getexif().get(0x0112, 1))
            rec["width_raw"], rec["height_raw"] = im.size
            img = ImageOps.exif_transpose(im).convert("RGB")
        rec["width"], rec["height"] = img.size
        rec["pixel_sha256"] = hashlib.sha256(img.tobytes()).hexdigest()
        # deterministic cache: EXIF-corrected RGB, shorter side -> short_side (bicubic), JPEG q95
        s = short_side / min(img.size)
        small = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC) if s < 1 else img
        for name, v in zip(VARIANTS, _variants(small)):  # pHash downsamples to 32x32 anyway
            rec[f"phash_{name}"] = str(imagehash.phash(v))
        out = Path(cache_dir) / rec["rel_path"]
        out = out.with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        small.save(out, quality=95)
        rec["cache_path"] = str(out.relative_to(cache_dir))
        rec["decode_status"] = "ok"
    except Exception as e:  # recorded per file, never silently skipped
        rec["decode_status"] = f"error: {type(e).__name__}: {e}"
    return rec


def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _contact_sheet(pairs: list[tuple], df: pd.DataFrame, cache: Path, out: Path, title: str) -> None:
    if not pairs:
        return
    tile, cols = 160, 2
    rows = len(pairs)
    sheet = Image.new("RGB", (cols * tile * 2 + 20, rows * (tile + 24) + 30), "white")
    d = ImageDraw.Draw(sheet)
    d.text((5, 5), title, fill="black")
    for r, (i, j, dist) in enumerate(pairs):
        y = 30 + r * (tile + 24)
        for c, k in enumerate((i, j)):
            im = Image.open(cache / df.at[k, "cache_path"]).convert("RGB")
            im.thumbnail((tile * 2 - 10, tile))
            sheet.paste(im, (c * tile * 2 + 5, y))
            d.text((c * tile * 2 + 5, y + tile + 2),
                   f"{df.at[k, 'label_original'][:22]} | {Path(df.at[k, 'rel_path']).name[:22]}", fill="black")
        d.text((cols * tile * 2 - 60, y + tile + 2), f"d={dist}", fill="red")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def run_audit(cfg: dict) -> None:
    d, a = cfg["data"], cfg["audit"]
    raw = Path(d["raw_dir"])
    cache = Path(d["cache_dir"])
    mdir = Path(d["manifest_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    source = json.loads((raw / "source_metadata.json").read_text()) if (raw / "source_metadata.json").exists() else {}
    root = find_image_root(raw / "original")
    log(f"image root: {root}")
    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    non_images = [str(p.relative_to(root)) for p in all_files if p.suffix.lower() not in IMG_EXT]
    paths = [p for p in all_files if p.suffix.lower() in IMG_EXT]
    log(f"found {len(paths)} image files ({len(non_images)} non-image files) in {len({p.parent for p in paths})} folders")

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        recs = list(tqdm(ex.map(_process, [(str(p), str(root), str(cache), d["cache_short_side"]) for p in paths],
                                chunksize=8), total=len(paths), desc="audit: decode+hash+cache", unit="img"))
    df = pd.DataFrame(recs)
    df["source"] = f"{source.get('doi', 'mendeley:' + d['mendeley_id'])}/v{d['version']}/original"
    df["empty_file"] = df["file_bytes"] == 0
    classes = sorted(df["label_original"].unique())
    class_map = [{"class_id": i, "folder_name": c, "display_name": c.replace("_", " ").strip()} for i, c in enumerate(classes)]
    df["class_id"] = df["label_original"].map({c: i for i, c in enumerate(classes)})
    ok = df["decode_status"] == "ok"
    log(f"decode ok: {ok.sum()} / {len(df)}; classes: {len(classes)}")

    # ---- duplicates -------------------------------------------------------
    n = len(df)
    uf_verified, uf_all = _UF(n), _UF(n)
    for col in ("sha256", "pixel_sha256"):
        for _, idx in df[ok].groupby(col).groups.items():
            idx = list(idx)
            for k in idx[1:]:
                uf_verified.union(idx[0], k)
                uf_all.union(idx[0], k)
    oki = np.flatnonzero(ok.values)
    H = {v: np.array([int(h, 16) for h in df.loc[oki, f"phash_{v}"]], dtype=np.uint64) for v in VARIANTS}
    thr = a["phash_near_dup_threshold"]
    near_pairs = {}
    for r0 in tqdm(range(0, len(oki), 256), desc="audit: near-duplicate search", unit="block"):
        rows = slice(r0, r0 + 256)
        best = None
        for v in VARIANTS:  # image i (identity) vs transformed variants of j -> transformed duplicates
            dist = _hamming_matrix(H["id"][rows], H[v])
            best = dist if best is None else np.minimum(best, dist)
        ii, jj = np.nonzero(best <= thr)
        for i, j in zip(ii + r0, jj):
            if i < j:
                key = (int(oki[i]), int(oki[j]))
                near_pairs[key] = min(near_pairs.get(key, 99), int(best[i - r0, j]))
    exact_pairs = set()
    for col in ("sha256", "pixel_sha256"):
        for _, idx in df[ok].groupby(col).groups.items():
            idx = sorted(idx)
            exact_pairs |= {(idx[0], k) for k in idx[1:]}
    heuristic_pairs = {k: v for k, v in near_pairs.items() if k not in exact_pairs}
    for i, j in heuristic_pairs:
        uf_all.union(i, j)
    df["verified_dup_group"] = [f"v{uf_verified.find(i):05d}" for i in range(n)]
    df["group_id"] = [f"g{uf_all.find(i):05d}" for i in range(n)]
    gsize = df.groupby("group_id")["rel_path"].transform("size")
    vsize = df.groupby("verified_dup_group")["rel_path"].transform("size")
    df["group_type"] = np.where(gsize == 1, "singleton",
                                np.where(gsize == vsize, "verified_exact_duplicate", "heuristic_phash_similarity"))
    df["group_provenance"] = np.where(
        df["group_type"] == "singleton", "no duplicate found",
        np.where(df["group_type"] == "verified_exact_duplicate", "identical file bytes or decoded pixels",
                 f"pHash Hamming <= {thr} incl. flips/rotations (heuristic; not a verified specimen ID)"))

    # ---- quarantine -------------------------------------------------------
    labels_per_group = df[ok].groupby("group_id")["label_original"].nunique()
    conflict_groups = set(labels_per_group[labels_per_group > 1].index)
    df["exclusion_reason"] = ""
    df.loc[~ok, "exclusion_reason"] = "decode_failure"
    df.loc[df["empty_file"], "exclusion_reason"] = "empty_file"
    # byte/pixel-identical copies are not independent originals: keep the first (by path) per verified group
    redundant = ok & df.duplicated("verified_dup_group", keep="first") & (vsize > 1)
    df.loc[redundant, "exclusion_reason"] = "redundant_exact_copy"
    df.loc[ok & df["group_id"].isin(conflict_groups), "exclusion_reason"] = "conflicting_label_duplicate_group"
    df["eligible"] = df["exclusion_reason"] == ""

    # ---- contact sheets for inspection -----------------------------------
    srt = sorted(heuristic_pairs.items(), key=lambda kv: kv[1])
    cross = [(i, j, dd) for (i, j), dd in srt if df.at[i, "label_original"] != df.at[j, "label_original"]]
    same = [(i, j, dd) for (i, j), dd in srt if df.at[i, "label_original"] == df.at[j, "label_original"]]
    m = a["contact_sheet_max_pairs"]
    sheets = mdir / "contact_sheets"
    _contact_sheet(cross[:m], df, cache, sheets / "near_dup_cross_label.png", "near-duplicate candidates, DIFFERENT labels")
    _contact_sheet(same[:m], df, cache, sheets / "near_dup_same_label.png", "near-duplicate candidates, same label (closest)")
    _contact_sheet(same[-m:][::-1] if len(same) > m else [], df, cache, sheets / "near_dup_same_label_borderline.png",
                   f"near-duplicate candidates at threshold edge (d<={thr})")
    ex_pairs = sorted(exact_pairs)[:m]
    _contact_sheet([(i, j, 0) for i, j in ex_pairs], df, cache, sheets / "exact_duplicates.png", "exact duplicates")

    pd.DataFrame([{"a": df.at[i, "rel_path"], "b": df.at[j, "rel_path"], "phash_dist": dd,
                   "label_a": df.at[i, "label_original"], "label_b": df.at[j, "label_original"],
                   "kind": "exact" if (i, j) in exact_pairs else "heuristic"}
                  for (i, j), dd in {**{p: 0 for p in exact_pairs}, **heuristic_pairs}.items()]
                 ).to_csv(mdir / "duplicate_pairs.csv", index=False)
    cols = ["rel_path", "source", "label_original", "class_id", "width", "height", "width_raw", "height_raw",
            "exif_orientation", "format", "mode_original", "file_bytes", "empty_file", "sha256", "pixel_sha256",
            "decode_status", "phash_id", "verified_dup_group", "group_id", "group_type", "group_provenance",
            "exclusion_reason", "eligible", "cache_path"]
    df[cols].to_csv(mdir / "image_manifest.csv", index=False)
    atomic_json(mdir / "class_map.json", class_map)

    counts = df.groupby("label_original").agg(images=("rel_path", "size"), eligible=("eligible", "sum"),
                                               groups=("group_id", "nunique")).reset_index()
    report = {
        "source": source, "image_root": str(root), "expected_total_stated_by_source": EXPECTED_TOTAL,
        "observed_total_images": int(len(df)), "matches_stated_total": int(len(df)) == EXPECTED_TOTAL,
        "non_image_files": non_images, "n_classes": len(classes),
        "decode_failures": int((~ok).sum()), "empty_files": int(df["empty_file"].sum()),
        "exif_orientation_counts": {str(k): int(v) for k, v in df["exif_orientation"].value_counts(dropna=False).items()},
        "mode_counts": {str(k): int(v) for k, v in df["mode_original"].value_counts().items()},
        "exact_duplicate_pairs": len(exact_pairs), "heuristic_near_duplicate_pairs": len(heuristic_pairs),
        "heuristic_cross_label_pairs": len(cross),
        "group_type_counts": {k: int(v) for k, v in df.drop_duplicates("group_id")["group_type"].value_counts().items()},
        "conflicting_label_groups": len(conflict_groups),
        "excluded_images": {k: int(v) for k, v in df.loc[~df["eligible"], "exclusion_reason"].value_counts().items()},
        "eligible_images": int(df["eligible"].sum()), "eligible_groups": int(df[df["eligible"]]["group_id"].nunique()),
        "per_class": counts.to_dict(orient="records"),
        "size_summary": {"min_wh": [int(df["width"].min()), int(df["height"].min())],
                         "max_wh": [int(df["width"].max()), int(df["height"].max())],
                         "distinct_sizes": int(df[["width", "height"]].drop_duplicates().shape[0])},
        "notes": [
            "No specimen/plant/capture metadata is provided by the source; groups are duplicate-screening groups only.",
            "Heuristic pHash groups are kept within one partition for leakage safety; they are not claimed to be the same specimen.",
        ],
    }
    atomic_json(mdir / "audit_report.json", report)
    log(f"observed {len(df)} images (source states {EXPECTED_TOTAL}); eligible {report['eligible_images']} "
        f"in {report['eligible_groups']} groups")
    log(f"exact-dup pairs {len(exact_pairs)}, heuristic near-dup pairs {len(heuristic_pairs)} "
        f"({len(cross)} cross-label); conflicting groups quarantined: {len(conflict_groups)}")
    print(counts.to_string(index=False))
    log(f"wrote {mdir}/image_manifest.csv, audit_report.json, class_map.json, contact_sheets/")

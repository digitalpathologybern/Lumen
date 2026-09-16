#!/usr/bin/env python3
"""Stream PathGen-1.6M into a local patch+caption corpus.

PathGen ships only captions plus WSI coordinates (see
``assets/datasets/pathgen/PathGen-1.6M.json``); the images live in TCGA. This
script reconstructs the image--text pairs without ever hoarding whole slides:
for each of the 7,272 referenced WSIs it downloads the ``.svs`` from the GDC
open-access data endpoint into scratch, cuts out only the ~223 patches PathGen
references, writes them as JPEGs plus a caption lookup, then deletes the slide.
Peak disk is a handful of in-flight slides, not the multi-TB TCGA subset.

Geometry follows PathGen's own released spec exactly: ``read_region((x, y), 0,
(672, 672))`` with ``position`` the level-0 top-left corner.

Output layout (drop-in for :class:`lumen.training.data.QuiltPairDataset`)::

    <out>/patches/<wsi_id>/<wsi_id>_<x>_<y>.jpg
    <out>/shards/<wsi_id>.csv          # image_path,caption  (one per WSI)
    <out>/done/<wsi_id>                # completion marker (resume)
    <out>/failures.log                 # wsi_id \t error

Merge the per-WSI shards into a single lookup CSV with ``--merge`` once the run
(or runs) finish. The job is resumable: re-running skips any WSI whose done
marker exists.

Needs OpenSlide, so run under the ``lumen`` env, e.g.::

    python \\
        cli/build_pathgen_patches.py --workers 16
"""

from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import openslide
from PIL import Image
from tqdm import tqdm

GDC_DATA = "https://api.gdc.cancer.gov/data/{file_id}"
PATCH = (672, 672)  # PathGen's released extraction size, at level 0


def _download(file_id: str, dest: Path, retries: int = 5) -> None:
    """Fetch one open-access slide to ``dest`` with curl (resume + retry)."""
    url = GDC_DATA.format(file_id=file_id)
    for attempt in range(1, retries + 1):
        r = subprocess.run(
            ["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "5",
             "-C", "-", "-o", str(dest), url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return
        time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def _extract_one(args) -> tuple[str, int, str | None]:
    """Download one WSI, cut its patches, write a shard, delete the slide.

    Returns (wsi_id, n_patches_written, error_or_None). Runs in a worker
    process; all paths are recomputed here to stay picklable and independent.
    """
    wsi_id, file_id, items, out, scratch, fmt, quality = args
    out = Path(out)
    scratch = Path(scratch)
    ext = "png" if fmt == "png" else "jpg"
    patch_dir = out / "patches" / wsi_id
    shard = out / "shards" / f"{wsi_id}.csv"
    done = out / "done" / wsi_id
    lock_path = out / "locks" / f"{wsi_id}.lock"
    svs = scratch / f"{file_id}.svs"

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        # The pending list is only a snapshot.  Serialize the check and the
        # download across workers on this node so they cannot fetch it twice.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if done.exists():
            return wsi_id, 0, None  # another worker or prior run finished it

        slide = None
        try:
            patch_dir.mkdir(parents=True, exist_ok=True)
            _download(file_id, svs)
            slide = openslide.OpenSlide(str(svs))
            rows = []
            for it in items:
                x, y = int(it["position"][0]), int(it["position"][1])
                fname = f"{wsi_id}_{x}_{y}.{ext}"
                fpath = patch_dir / fname
                if not fpath.exists():
                    region = slide.read_region((x, y), 0, PATCH).convert("RGB")
                    if fmt == "png":
                        region.save(fpath, "PNG")  # lossless: no compression artifacts
                    else:
                        region.save(fpath, "JPEG", quality=quality)
                # store path relative to <out> so the corpus is relocatable
                rows.append((str(fpath.relative_to(out)), it["caption"]))
            with open(shard, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["image_path", "caption"])
                w.writerows(rows)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text("")
            return wsi_id, len(rows), None
        except Exception as e:  # noqa: BLE001 - one bad slide must not kill the run
            return wsi_id, 0, f"{type(e).__name__}: {e}"
        finally:
            if slide is not None:
                slide.close()
            try:
                svs.unlink(missing_ok=True)  # never keep the whole slide
            except OSError:
                pass


def _group(records) -> "collections.OrderedDict[str, tuple[str, list]]":
    """Group entries by WSI so each slide is downloaded exactly once."""
    g: "collections.OrderedDict[str, tuple[str, list]]" = collections.OrderedDict()
    for r in records:
        wsi = r["wsi_id"]
        if wsi not in g:
            g[wsi] = (r["file_id"], [])
        g[wsi][1].append(r)
    return g


def _merge(out: Path) -> None:
    """Concatenate per-WSI shards into a single training lookup CSV."""
    lookup = out / "pathgen_lookup.csv"
    shards = sorted((out / "shards").glob("*.csv"))
    n = 0
    with open(lookup, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "caption"])
        for s in shards:
            with open(s, newline="") as sh:
                rd = csv.reader(sh)
                next(rd, None)  # header
                for row in rd:
                    w.writerow(row)
                    n += 1
    print(f"merged {len(shards)} shards -> {lookup}  ({n} pairs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[1] / "assets" / "datasets" / "pathgen"
    ap.add_argument("--json", type=Path, default=default_root / "PathGen-1.6M.json")
    ap.add_argument("--out", type=Path, default=default_root)
    ap.add_argument("--scratch", type=Path,
                    default=Path(os.environ.get("TMPDIR", "/tmp")) / "pathgen_svs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--format", choices=["png", "jpeg"], default="jpeg",
                    help="patch encoding: jpeg (default, q95) or png (lossless)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="only used when --format jpeg")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N WSIs (smoke test)")
    ap.add_argument("--merge", action="store_true",
                    help="just concatenate finished shards into the lookup CSV and exit")
    args = ap.parse_args()

    for sub in ("patches", "shards", "done"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(args.out)
        return

    print(f"loading {args.json} ...", flush=True)
    records = json.load(open(args.json))
    groups = _group(records)
    wsis = list(groups.items())
    if args.limit:
        wsis = wsis[: args.limit]
    total = len(wsis)
    pending = [(w, fid, items, str(args.out), str(args.scratch), args.format,
                args.jpeg_quality)
               for w, (fid, items) in wsis if not (args.out / "done" / w).exists()]
    print(f"{total} WSIs in scope; {len(pending)} pending, "
          f"{total - len(pending)} already done", flush=True)

    fail = open(args.out / "failures.log", "a")
    already = total - len(pending)  # finished on prior runs; bar counts overall progress
    done_n = patches_n = err_n = 0
    # Progress bar to stdout so it lands in the SLURM .log. mininterval keeps the
    # log from being spammed; the count is over ALL 7,272 WSIs, not just this run.
    bar = tqdm(total=total, initial=already, unit="wsi", file=sys.stdout,
               mininterval=10, dynamic_ncols=True, smoothing=0.05,
               desc="PathGen WSIs")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_extract_one, p): p[0] for p in pending}
        for fut in as_completed(futs):
            wsi_id, n, err = fut.result()
            done_n += 1
            if err:
                err_n += 1
                fail.write(f"{wsi_id}\t{err}\n")
                fail.flush()
            else:
                patches_n += n
            bar.update(1)
            bar.set_postfix(patches=patches_n, failed=err_n, refresh=False)
    bar.close()
    fail.close()
    print(f"finished this run: {done_n} slides, {patches_n} patches, {err_n} failures; "
          f"{already + done_n - err_n}/{total} WSIs complete overall", flush=True)
    print("run with --merge to build pathgen_lookup.csv once all shards are present")


if __name__ == "__main__":
    main()

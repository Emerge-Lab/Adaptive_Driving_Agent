"""Build chunked symlink map dirs for full-pool per-map scoring, plus the
genuinely-held-out test dir (original ids 4999..5401, never sampled in training
since binding.h draws map_id = rand() % num_maps with num_maps=4999).

Outputs:
  resources/drive/binaries/nuplan_201_chunk{00..09}/map_%03d.bin  (symlinks)
  resources/drive/binaries/nuplan_heldout_403/map_%03d.bin        (symlinks)
  outputs/map_scoring/chunk_manifest.csv   (chunk,local_id,orig_id)
"""
import csv
from pathlib import Path

SRC = Path("resources/drive/binaries/nuplan_201")
CHUNK_ROOT = Path("resources/drive/binaries")
MANIFEST = Path("outputs/map_scoring/chunk_manifest.csv")
CHUNK_SIZE = 541
HELDOUT_START = 4999


def link(dst_dir: Path, local_id: int, orig_id: int):
    dst = dst_dir / f"map_{local_id:03d}.bin"
    src = (SRC / f"map_{orig_id:03d}.bin").resolve()
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    n_maps = len(list(SRC.glob("map_*.bin")))
    print(f"source maps: {n_maps}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_chunks = (n_maps + CHUNK_SIZE - 1) // CHUNK_SIZE
    for c in range(n_chunks):
        d = CHUNK_ROOT / f"nuplan_201_chunk{c:02d}"
        d.mkdir(exist_ok=True)
        lo, hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n_maps)
        for local_id, orig_id in enumerate(range(lo, hi)):
            link(d, local_id, orig_id)
            rows.append({"chunk": c, "local_id": local_id, "orig_id": orig_id})
        print(f"chunk{c:02d}: {hi - lo} maps (orig {lo}..{hi - 1})")

    d = CHUNK_ROOT / "nuplan_heldout_403"
    d.mkdir(exist_ok=True)
    for local_id, orig_id in enumerate(range(HELDOUT_START, n_maps)):
        link(d, local_id, orig_id)
    print(f"nuplan_heldout_403: {n_maps - HELDOUT_START} maps (orig {HELDOUT_START}..{n_maps - 1})")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chunk", "local_id", "orig_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {MANIFEST} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

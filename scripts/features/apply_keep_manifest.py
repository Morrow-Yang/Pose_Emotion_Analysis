import argparse
import os
import shutil
from pathlib import Path


def hardlink_or_copy(src: Path, dst: Path, hardlink: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Apply keep manifest to create filtered subset")
    ap.add_argument("--root", required=True, help="root of original frames, e.g., outputs/frames/CAER/train")
    ap.add_argument("--keep", required=True, help="keep manifest file")
    ap.add_argument("--out", required=True, help="output root for filtered frames")
    ap.add_argument("--hardlink", action="store_true", help="use hard links instead of copying")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    keep_path = Path(args.keep)
    if not keep_path.exists():
        raise SystemExit(f"keep manifest not found: {keep_path}")

    lines = keep_path.read_text(encoding="utf-8").strip().splitlines()
    total = len(lines)
    print(f"Applying manifest: {total} entries")

    for i, rel in enumerate(lines, 1):
        src = root / rel
        dst = out_root / rel
        if not src.exists():
            print(f"[WARN] missing: {src}")
            continue
        hardlink_or_copy(src, dst, hardlink=args.hardlink)
        if i % 5000 == 0:
            print(f"  processed {i}/{total}")

    print("Done.")


if __name__ == "__main__":
    main()
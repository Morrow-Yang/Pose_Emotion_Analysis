import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from ultralytics import YOLO


def load_images(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])


def should_keep(boxes, img_w: int, img_h: int, min_area_ratio: float, min_height_ratio: float,
                max_ar: float) -> bool:
    if boxes is None or len(boxes) == 0:
        return False
    img_area = float(img_w * img_h)
    min_area = img_area * min_area_ratio
    min_h = img_h * min_height_ratio
    for b in boxes:
        x1, y1, x2, y2 = b
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w <= 0 or h <= 0:
            continue
        area = w * h
        ar = max(w / h, h / w)
        if area >= min_area and h >= min_h and ar <= max_ar:
            return True
    return False


def filter_images(model: YOLO, images: List[Path], conf: float, min_area_ratio: float,
                  min_height_ratio: float, max_ar: float, batch: int) -> Dict[str, bool]:
    kept = {}
    for i in range(0, len(images), batch):
        batch_paths = images[i:i + batch]
        results = model(batch_paths, conf=conf, classes=[0], verbose=False)
        for path, res in zip(batch_paths, results):
            im0 = res.orig_img
            h, w = im0.shape[:2]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            kept[str(path)] = should_keep(boxes, w, h, min_area_ratio, min_height_ratio, max_ar)
    return kept


def write_manifest(kept: Dict[str, bool], root: Path, out_keep: Path, out_drop: Path) -> None:
    keep_list = [str(Path(k).relative_to(root)) for k, v in kept.items() if v]
    drop_list = [str(Path(k).relative_to(root)) for k, v in kept.items() if not v]
    out_keep.parent.mkdir(parents=True, exist_ok=True)
    out_drop.parent.mkdir(parents=True, exist_ok=True)
    out_keep.write_text('\n'.join(keep_list), encoding='utf-8')
    out_drop.write_text('\n'.join(drop_list), encoding='utf-8')
    print(f"kept {len(keep_list)} / {len(kept)} images")
    print(f"keep list -> {out_keep}")
    print(f"drop list -> {out_drop}")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 human filter for CAER frames")
    ap.add_argument('--root', required=True, help='root of extracted frames, e.g., outputs/frames/CAER/train')
    ap.add_argument('--pattern', nargs='*', default=['*.jpg', '*.png'], help='glob patterns to include')
    ap.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model weights')
    ap.add_argument('--conf', type=float, default=0.35, help='confidence threshold')
    ap.add_argument('--min_area_ratio', type=float, default=0.01, help='min bbox area ratio to image')
    ap.add_argument('--min_height_ratio', type=float, default=0.1, help='min bbox height ratio to image')
    ap.add_argument('--max_ar', type=float, default=4.0, help='max aspect ratio (long/short) to avoid extreme skinny boxes')
    ap.add_argument('--batch', type=int, default=16, help='batch size for inference')
    ap.add_argument('--out_keep', default='outputs/manifests/caer_keep.txt')
    ap.add_argument('--out_drop', default='outputs/manifests/caer_drop.txt')
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    exts = tuple(set([Path(p).suffix.lower() for p in args.pattern]))
    images = load_images(root, exts)
    if not images:
        raise SystemExit("no images found under root with given pattern")

    print(f"loaded {len(images)} images from {root}")
    model = YOLO(args.model)

    kept = filter_images(model, images, args.conf, args.min_area_ratio, args.min_height_ratio, args.max_ar, args.batch)
    write_manifest(kept, root, Path(args.out_keep), Path(args.out_drop))


if __name__ == '__main__':
    main()

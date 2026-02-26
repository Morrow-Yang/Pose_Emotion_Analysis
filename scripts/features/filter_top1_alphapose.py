import json
from pathlib import Path
import argparse


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    if isinstance(data, dict):
        # openpose-style: {image: {people: [{pose_keypoints_2d: [...]}, ...]}}
        for img, val in data.items():
            people = []
            if isinstance(val, dict):
                people = val.get("people", []) or []
            for p in people:
                kp = p.get("pose_keypoints_2d", [])
                records.append({
                    "image_id": img,
                    "keypoints": kp,
                    "score": p.get("score", 1.0),
                    "box": p.get("box", [0, 0, 1, 1]),
                })
    elif isinstance(data, list):
        records = data
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(inp)
    if not records:
        print("[WARN] no records loaded", inp)
        json.dump([], outp.open("w", encoding="utf-8"))
        return

    # group by image_id
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r.get("image_id", "")] .append(r)

    kept = []
    for img, lst in groups.items():
        # choose by score then bbox area
        def keyfn(x):
            box = x.get("box", [0, 0, 0, 0])
            area = box[2] * box[3] if len(box) == 4 else 0
            return (x.get("score", 0), area)
        best = max(lst, key=keyfn)
        kept.append(best)

    json.dump(kept, outp.open("w", encoding="utf-8"))
    print(f"[OK] kept {len(kept)} frames -> {outp}")


if __name__ == "__main__":
    main()

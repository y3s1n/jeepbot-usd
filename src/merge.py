"""merge.py - rebuild labels/ and split.csv from every Label Studio export.

    python src/merge.py
    python src/merge.py --check history/split_reference.csv    # do this after any edit

Reads:
    <root>/dataset/<folder>/<folder>_<frame>.jpg   the master dataset, all frames
    <root>/exports/*.json                           Label Studio exports, never edited
Writes:
    <root>/labels/<folder>/<frame>.txt              one line per box: class cx cy w h
    <root>/split.csv                                every labelled frame and its split

Always rebuilds from scratch, so run it again whenever a new export arrives.

The split rule and the rotated-box handling below are ported unchanged from the
predecessor on purpose (ADR 0004 decision 4). A silently different split
invalidates every number in this repository without raising an error, which is
what --check exists to catch: it diffs the split.csv just written against the
committed reference and exits non-zero on any difference.
"""
import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

# Must match the label names in Label Studio exactly. An unknown name stops the script.
CLASSES = {"person": 0, "fire_hydrant": 1, "trash_can": 2, "ignore": -1}
# "ignore" = a real object too small, dark or hidden to learn from. Written as class -1:
# training neither rewards nor punishes the network there, and scoring skips it.
# Barely used by the original labelling pass: 2.6% of boxes. See ADR 0004.

VAL_FRACTION = 0.20   # last 20% of each folder, by frame number, is validation
BUFFER = 10           # frames dropped each side of the train/val boundary
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ROTATED = []          # boxes that were rotated in Label Studio and got straightened

PATH_RE = re.compile(r"dataset/([^/]+)/([^/?&\"']+\.(?:jpg|jpeg|png))", re.IGNORECASE)
FRAME_RE = re.compile(r"_(\d+)\.\w+$")


def frame_number(filename):
    m = FRAME_RE.search(filename)
    if not m:
        raise ValueError(f"No frame number in filename: {filename}")
    return int(m.group(1))


def build_split_rule(dataset_dir):
    """Assign every frame in the master dataset to train, val, or buffer.

    Based on ALL frames in each folder, not just the labelled ones, so the rule
    never shifts as more labels arrive. Consecutive frames from a moving vehicle
    are near duplicates, so this is positional rather than random on purpose.
    """
    rule = {}
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        frames = sorted(
            (f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT),
            key=lambda f: frame_number(f.name),
        )
        cut = int(len(frames) * (1 - VAL_FRACTION))
        for i, f in enumerate(frames):
            if cut - BUFFER <= i < cut + BUFFER:
                split = "buffer"
            elif i < cut:
                split = "train"
            else:
                split = "val"
            rule[(folder.name, f.name)] = split
    return rule


def image_key(task, source):
    """Find (folder, filename) from the task's image path."""
    text = unquote(json.dumps(task.get("data", {})))
    m = PATH_RE.search(text)
    if not m:
        raise ValueError(f"No dataset/<folder>/<file> path in task {task.get('id')} ({source})")
    return m.group(1), m.group(2)


def pick_annotation(task):
    """Latest submitted, non-skipped annotation, or None."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled", False)]
    if not anns:
        return None
    return max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def boxes_from(annotation, where):
    """Label Studio rectangles (percent, top-left) to class cx cy w h (fractions)."""
    boxes = []
    for r in annotation.get("result", []):
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        names = v.get("rectanglelabels", [])
        if len(names) != 1:
            raise ValueError(f"Box with {len(names)} labels in {where}")
        name = names[0]
        if name not in CLASSES:
            raise ValueError(f"Unknown label '{name}' in {where}. Expected one of {list(CLASSES)}")
        rotation = v.get("rotation", 0) or 0
        if abs(rotation) > 1e-6:
            # Label Studio rotates clockwise around the top-left corner. Replace the rotated
            # box with the smallest upright box enclosing it (in pixels, then back to %).
            W, H = r.get("original_width", 1280), r.get("original_height", 800)
            x0, y0 = v["x"] / 100 * W, v["y"] / 100 * H
            bw, bh = v["width"] / 100 * W, v["height"] / 100 * H
            t = math.radians(rotation)
            xs, ys = [], []
            for dx, dy in [(0, 0), (bw, 0), (bw, bh), (0, bh)]:
                xs.append(x0 + dx * math.cos(t) - dy * math.sin(t))
                ys.append(y0 + dx * math.sin(t) + dy * math.cos(t))
            left, right = max(min(xs), 0), min(max(xs), W)
            top, bottom = max(min(ys), 0), min(max(ys), H)
            w, h = (right - left) / W, (bottom - top) / H
            cx, cy = (left + right) / 2 / W, (top + bottom) / 2 / H
            ROTATED.append(where)
        else:
            w = v["width"] / 100
            h = v["height"] / 100
            cx = v["x"] / 100 + w / 2
            cy = v["y"] / 100 + h / 2
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        boxes.append((CLASSES[name], cx, cy, w, h))
    return boxes


def compare(written, reference):
    """Print the first differences between two split.csv files. True if identical."""
    a = Path(written).read_text(encoding="utf-8").splitlines()
    b = Path(reference).read_text(encoding="utf-8").splitlines()
    if a == b:
        print(f"check: split.csv matches {reference} exactly ({len(a) - 1} rows)")
        return True
    print(f"check: DIFFERS from {reference}.  written {len(a) - 1} rows, reference {len(b) - 1}")
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    for label, rows in (("only in written", only_a), ("only in reference", only_b)):
        for row in sorted(rows)[:5]:
            print(f"  {label}: {row}")
        if len(rows) > 5:
            print(f"  {label}: ... and {len(rows) - 5} more")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--check", metavar="SPLIT_CSV",
                    help="diff the split.csv written against this reference, exit 1 if different")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    dataset_dir = root / "dataset"
    exports_dir = root / "exports"
    labels_dir = root / "labels"

    rule = build_split_rule(dataset_dir)

    seen = {}      # (folder, file) -> export filename
    rows = []
    counts = {"skipped_or_unlabeled": 0, "buffer": 0, "train": 0, "val": 0}
    class_counts = {s: {c: 0 for c in CLASSES} for s in ("train", "val")}
    staged = {}    # (folder, file) -> boxes

    exports = sorted(exports_dir.glob("*.json"))
    if not exports:
        raise SystemExit(f"No exports in {exports_dir}")

    names = {v: k for k, v in CLASSES.items()}
    for export in exports:
        tasks = json.loads(export.read_text(encoding="utf-8"))
        for task in tasks:
            key = image_key(task, export.name)
            if key in seen:
                raise ValueError(f"{key[1]} appears in both {seen[key]} and {export.name}")
            seen[key] = export.name

            ann = pick_annotation(task)
            if ann is None:
                counts["skipped_or_unlabeled"] += 1
                continue
            if key not in rule:
                raise ValueError(f"{key} is not in the master dataset folder")

            split = rule[key]
            if split == "buffer":
                counts["buffer"] += 1
                continue

            boxes = boxes_from(ann, f"{key[1]} ({export.name})")
            staged[key] = boxes
            counts[split] += 1
            for c, *_ in boxes:
                class_counts[split][names[c]] += 1
            rows.append([key[0], key[1], split, len(boxes), export.name])

    # Only touch labels/ once every export has parsed cleanly.
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    for (folder, file), boxes in staged.items():
        out = labels_dir / folder / (Path(file).stem + ".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                               for c, cx, cy, w, h in boxes))

    split_csv = root / "split.csv"
    with open(split_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "file", "split", "n_boxes", "export"])
        writer.writerows(sorted(rows))

    print(f"Exports read: {len(exports)}")
    print(f"Frames  train: {counts['train']}  val: {counts['val']}  "
          f"buffer (dropped): {counts['buffer']}  skipped/unlabeled: {counts['skipped_or_unlabeled']}")
    for split in ("train", "val"):
        print(f"Boxes in {split}: " + "  ".join(f"{k}: {v}" for k, v in class_counts[split].items()))
    if ROTATED:
        print(f"Straightened {len(ROTATED)} rotated box(es). First few:")
        for where in ROTATED[:5]:
            print("   ", where)

    if args.check and not compare(split_csv, args.check):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

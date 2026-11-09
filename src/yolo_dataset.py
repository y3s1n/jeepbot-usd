"""yolo_dataset.py - build an ultralytics dataset from split.csv and labels/.

    python src/yolo_dataset.py                                   # person folders only
    python src/yolo_dataset.py --prefix person empty --out yolo_pe --val-prefix person

Writes the layout ultralytics expects:

    <out>/images/train/<stem>.jpg   symlink to <root>/dataset/<folder>/<file>
    <out>/labels/train/<stem>.txt   one line per kept box: 0 cx cy w h
    <out>/images/val/...            same, for the val rows
    <out>/data.yaml

Images are symlinked, never copied. Labels are rewritten because only one class
survives: person stays, ignore (-1) is DROPPED so a detection there counts as a
false alarm, and every other class is dropped (ADR 0003 decision 2). A frame
left with no boxes becomes an empty label file, which ultralytics trains on as
a background image.

--val-prefix exists for the experiment in ADR 0003's open questions: training
on person+empty while validating on the person-only rows keeps the number
comparable to the recorded 0.716, because only the training data changed.
"""
import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PERSON_CLASS = 0
IGNORE_CLASS = -1


def rows_for(root, prefixes, split):
    """split.csv rows in this split whose folder starts with one of the prefixes."""
    with open(Path(root) / "split.csv", newline="") as f:
        return [r for r in csv.DictReader(f)
                if r["split"] == split and r["folder"].startswith(tuple(prefixes))]


def keep_boxes(label_path):
    """Person lines only, as '0 cx cy w h'. Returns (lines, n_ignore, n_other)."""
    kept, n_ignore, n_other = [], 0, 0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls == IGNORE_CLASS:
            n_ignore += 1
        elif cls == PERSON_CLASS:
            kept.append(f"{PERSON_CLASS} " + " ".join(parts[1:5]))
        else:
            n_other += 1
    return kept, n_ignore, n_other


def build(root, out, prefixes, val_prefixes):
    root, out = Path(root), Path(out)
    rows = {"train": rows_for(root, prefixes, "train"),
            "val": rows_for(root, val_prefixes, "val")}
    for split, got in rows.items():
        if not got:
            raise SystemExit(f"No {split} rows in split.csv match "
                             f"{prefixes if split == 'train' else val_prefixes}")
    if out.exists():
        shutil.rmtree(out)                      # always rebuild from scratch

    # A frame can sit in a folder it is not named after, so check the stems really
    # are unique across both splits before flattening them into one directory.
    stems = [Path(r["file"]).stem for got in rows.values() for r in got]
    if len(set(stems)) != len(stems):
        dupes = sorted({s for s in stems if stems.count(s) > 1})
        raise SystemExit(f"Duplicate frame names across folders, cannot flatten: {dupes[:5]}")

    totals = {}
    for split, got in rows.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        n_img = n_box = n_bg = n_ignore = n_other = 0
        for r in got:
            stem = Path(r["file"]).stem
            image = root / "dataset" / r["folder"] / r["file"]
            label = root / "labels" / r["folder"] / f"{stem}.txt"
            if not image.exists():
                raise SystemExit(f"Missing image {image}")
            if not label.exists():
                raise SystemExit(f"Missing label {label}")
            (out / "images" / split / f"{stem}.jpg").symlink_to(image)
            kept, n_i, n_o = keep_boxes(label)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(kept) + ("\n" if kept else ""))
            n_img += 1
            n_box += len(kept)
            n_bg += not kept
            n_ignore += n_i
            n_other += n_o
        totals[split] = (n_img, n_box, n_bg, n_ignore, n_other)

    (out / "data.yaml").write_text(
        f"# built by src/yolo_dataset.py from split.csv -- do not edit by hand\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n  {PERSON_CLASS}: person\n")
    return totals


def selftest():
    """Covers the box filter and the per-split prefix selection, which is the part
    that changed from the predecessor. The symlink layout is only exercisable on a
    filesystem that allows symlinks, so it is checked by running for real on Linux."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "lbl.txt").write_text("0 .5 .5 .1 .2\n2 .1 .1 .1 .1\n-1 .9 .9 .0 .0\n"
                                   "0 .2 .2 .1 .1 0.97\n  \n")
        kept, n_ignore, n_other = keep_boxes(d / "lbl.txt")
        assert kept == ["0 .5 .5 .1 .2", "0 .2 .2 .1 .1"], kept   # trailing score dropped
        assert (n_ignore, n_other) == (1, 1), (n_ignore, n_other)

        (d / "split.csv").write_text(
            "folder,file,split,n_boxes,export\n"
            "person_A_up_sun,a.jpg,train,2,e.json\n"
            "person_A_up_sun,b.jpg,val,1,e.json\n"
            "empty_B_up_sun,c.jpg,train,1,e.json\n"
            "empty_B_up_sun,d.jpg,val,3,e.json\n")
        assert [r["file"] for r in rows_for(d, ["person"], "train")] == ["a.jpg"]
        assert [r["file"] for r in rows_for(d, ["person"], "val")] == ["b.jpg"]
        # the experiment shape: train on both, validate on person only
        assert len(rows_for(d, ["person", "empty"], "train")) == 2
        assert [r["file"] for r in rows_for(d, ["person"], "val")] == ["b.jpg"]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--selftest", action="store_true", help="check the box filter and exit")
    ap.add_argument("--out", default=None, help="output dir (default <root>/yolo_ft)")
    ap.add_argument("--prefix", nargs="+", default=["person"],
                    help="folder prefixes for the TRAIN rows (default: person)")
    ap.add_argument("--val-prefix", nargs="+", default=None,
                    help="folder prefixes for the VAL rows (default: same as --prefix)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    out = Path(args.out) if args.out else Path(args.root) / "yolo_ft"
    totals = build(args.root, out, args.prefix, args.val_prefix or args.prefix)

    print(f"{'split':<7}{'images':>8}{'person boxes':>14}{'background':>12}"
          f"{'ignore dropped':>16}{'other dropped':>15}")
    for split, (n_img, n_box, n_bg, n_ignore, n_other) in totals.items():
        print(f"{split:<7}{n_img:>8}{n_box:>14}{n_bg:>12}{n_ignore:>16}{n_other:>15}")
    print(f"\nWrote {out}/data.yaml  (images are symlinks)")


if __name__ == "__main__":
    main()

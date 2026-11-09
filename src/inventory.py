"""inventory.py - read-only report on what is actually on the data machine.

    python src/inventory.py                  # expects dataset/ labels/ etc. at the repo root
    python src/inventory.py --root /data/jeepbot
    python src/inventory.py --selftest       # checks the label parser, touches no data

Writes nothing, deletes nothing, opens no images. Standard library only, so it
runs on whatever Python the data machine already has.

Slice 3 of the rebuild (ADR 0005): every delete decision waits on this output.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {0: "person", 1: "fire_hydrant", 2: "trash_can", -1: "ignore"}


def mb(n):
    return f"{n / 1e6:9.1f} MB"


def head(title):
    print(f"\n{title}\n{'-' * len(title)}")


def count_boxes(folder):
    """(label files, boxes per class id, files with no boxes) for one labels/ folder.

    Unknown class ids are counted under their own id rather than dropped: a
    stray id is exactly the kind of thing this report exists to surface.
    """
    classes, empty, files = Counter(), 0, 0
    for txt in folder.glob("*.txt"):
        files += 1
        n = 0
        for line in txt.read_text().splitlines():
            parts = line.split()
            if parts:
                classes[int(float(parts[0]))] += 1
                n += 1
        if not n:
            empty += 1
    return files, classes, empty


def report_dataset(root):
    head("dataset/ : captured frames")
    d = root / "dataset"
    if not d.is_dir():
        return print(f"absent: {d}")
    total_n = total_size = 0
    print(f"{'folder':<26}{'frames':>8}{'size':>13}")
    for folder in sorted(p for p in d.iterdir() if p.is_dir()):
        files = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT]
        size = sum(f.stat().st_size for f in files)
        total_n += len(files)
        total_size += size
        print(f"{folder.name:<26}{len(files):>8}{mb(size):>13}")
    print(f"{'TOTAL':<26}{total_n:>8}{mb(total_size):>13}")


def report_labels(root):
    head("labels/ : boxes per class, per folder")
    d = root / "labels"
    if not d.is_dir():
        return print(f"absent: {d}  (rebuild it from exports/)")
    ids, rows, totals = set(), [], Counter()
    for folder in sorted(p for p in d.iterdir() if p.is_dir()):
        files, classes, empty = count_boxes(folder)
        ids |= set(classes)
        totals.update(classes)
        totals["files"] += files
        totals["empty"] += empty
        rows.append((folder.name, files, empty, classes))
    order = sorted(ids, key=lambda c: (c not in CLASS_NAMES, c))
    names = [CLASS_NAMES.get(c, f"class {c}?") for c in order]
    print(f"{'folder':<26}{'files':>7}{'no boxes':>10}" + "".join(f"{n:>14}" for n in names))
    for name, files, empty, classes in rows:
        print(f"{name:<26}{files:>7}{empty:>10}" + "".join(f"{classes[c]:>14}" for c in order))
    print(f"{'TOTAL':<26}{totals['files']:>7}{totals['empty']:>10}"
          + "".join(f"{totals[c]:>14}" for c in order))
    unknown = [c for c in order if c not in CLASS_NAMES]
    if unknown:
        print(f"\nUNEXPECTED class ids present: {unknown}")


def report_exports(root):
    head("exports/ : Label Studio exports (the irreplaceable asset)")
    d = root / "exports"
    if not d.is_dir():
        return print(f"absent: {d}")
    print(f"{'file':<52}{'size':>13}{'tasks':>8}{'annotated':>11}")
    for f in sorted(d.glob("*.json")):
        size = f.stat().st_size
        try:
            tasks = json.loads(f.read_text(encoding="utf-8"))
            n = len(tasks)
            done = sum(1 for t in tasks
                       if any(not a.get("was_cancelled") for a in t.get("annotations", [])))
            print(f"{f.name:<52}{mb(size):>13}{n:>8}{done:>11}")
        except Exception as e:                       # a bad export must not kill the report
            print(f"{f.name:<52}{mb(size):>13}   unreadable: {type(e).__name__}")


def report_weights_and_runs(root):
    head("weights and training runs")
    seen = {}
    for pat in ("*.pt", "*.pth", "*.engine", "*.onnx", "checkpoints/*", "runs/**/weights/*.pt"):
        for f in root.glob(pat):
            if f.is_file():
                seen[f] = f.stat().st_size
    for f, size in sorted(seen.items()):
        print(f"{str(f.relative_to(root)):<52}{mb(size):>13}")
    if not seen:
        print("no weights found")
    for csv in sorted(root.glob("runs/**/results.csv")):
        epochs = max(sum(1 for _ in csv.open(encoding="utf-8", errors="replace")) - 1, 0)
        print(f"{str(csv.parent.relative_to(root)):<52}{epochs:>8} epochs")


def report_loose_files(root):
    head("loose data files at the root")
    for f in sorted(root.glob("*")):
        if f.is_file() and f.suffix.lower() in {".csv", ".json", ".yaml", ".txt"}:
            lines = sum(1 for _ in f.open(encoding="utf-8", errors="replace"))
            print(f"{f.name:<40}{mb(f.stat().st_size):>13}{lines:>9} lines")


def report_environments():
    head("python environments")
    print(f"running: {sys.executable}  ({sys.version.split()[0]})")
    try:
        out = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=30)
        print(out.stdout.strip() or "conda returned nothing")
    except Exception as e:
        print(f"conda not available ({type(e).__name__})")
    # bounded depth on purpose: rglob over a home directory can take minutes
    found = [cfg.parent for pat in ("*/pyvenv.cfg", "*/*/pyvenv.cfg", "*/*/*/pyvenv.cfg")
             for cfg in Path.home().glob(pat)]
    print("\nvirtualenvs under $HOME (depth 3): " + (", ".join(str(p) for p in found) or "none"))


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.txt").write_text("0 .5 .5 .1 .2\n0 .1 .1 .1 .1\n2 .3 .3 .1 .1\n")
        (d / "b.txt").write_text("-1 .5 .5 .1 .2\n")
        (d / "c.txt").write_text("")
        (d / "d.txt").write_text("\n  \n")                 # blank lines are not boxes
        (d / "e.txt").write_text("7.0 .5 .5 .1 .1\n")      # float id, unknown class
        files, classes, empty = count_boxes(d)
        assert files == 5, files
        assert empty == 2, empty                           # c.txt and d.txt
        assert dict(classes) == {0: 2, 2: 1, -1: 1, 7: 1}, dict(classes)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="dir holding dataset/, labels/, exports/")
    ap.add_argument("--selftest", action="store_true", help="check the label parser and exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    root = Path(args.root).expanduser().resolve()
    print(f"inventory of {root}")
    report_dataset(root)
    report_labels(root)
    report_exports(root)
    report_weights_and_runs(root)
    report_loose_files(root)
    report_environments()


if __name__ == "__main__":
    main()

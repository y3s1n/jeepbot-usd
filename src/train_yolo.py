"""train_yolo.py - fine-tune a YOLO model on the campus person frames.

    python src/yolo_dataset.py && python src/train_yolo.py

The defaults are the configuration that recorded 0.716 person mAP@0.5 in
ADR 0003: yolov8s.pt, imgsz 640, 100 epochs, batch 16, otherwise ultralytics
defaults. Changing them invalidates that number, which is why they are
constants here rather than something you have to remember to pass.

0.716 is the floor, not the target. --model and --imgsz are the two axes the
sweep explores; see ADR 0003's open questions.

Writes:
    <root>/runs/yolo_ft/<name>/weights/best.pt
    <root>/runs/yolo_ft/<name>/results.csv
Then re-validates best.pt and prints person mAP@0.5 with per-image latency.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ADR 0003 baseline.
MODEL = "yolov8s.pt"
IMGSZ = 640
EPOCHS = 100
BATCH = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--data", default=None, help="data.yaml (default <root>/yolo_ft/data.yaml)")
    ap.add_argument("--name", default=None, help="run name (default from model and imgsz)")
    ap.add_argument("--model", default=MODEL, help=f"pretrained weights (default {MODEL})")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    from ultralytics import YOLO          # imported late: slow, and only needed here

    root = Path(args.root)
    data = Path(args.data) if args.data else root / "yolo_ft" / "data.yaml"
    if not data.exists():
        raise SystemExit(f"No dataset at {data}. Run: python src/yolo_dataset.py")
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"

    weights = root / args.model
    if not weights.exists():
        print(f"{weights} not found; ultralytics will download {args.model}")
        weights = args.model

    print(f"model {args.model}  imgsz {args.imgsz}  epochs {args.epochs}  batch {args.batch}")
    YOLO(str(weights)).train(data=str(data), imgsz=args.imgsz, epochs=args.epochs,
                             batch=args.batch, project=str(root / "runs" / "yolo_ft"),
                             name=name, exist_ok=True)

    best = root / "runs" / "yolo_ft" / name / "weights" / "best.pt"
    print(f"\nre-validating {best}")
    results = YOLO(str(best)).val(data=str(data), imgsz=args.imgsz, split="val", verbose=False)

    # speed is per-image milliseconds on THIS machine, not on the Jetson. It is here so
    # the sweep has a cost axis to sort by; the real budget is measured on the board.
    ms = sum(results.speed.values())
    print("\n" + "=" * 52)
    print(f"{'run':<26}{name:>26}")
    print(f"{'person mAP@0.5':<26}{results.box.map50:>26.4f}")
    print(f"{'person mAP@0.5-0.95':<26}{results.box.map:>26.4f}")
    print(f"{'precision':<26}{results.box.mp:>26.4f}")
    print(f"{'recall':<26}{results.box.mr:>26.4f}")
    print(f"{'ms/image (this machine)':<26}{ms:>26.1f}")
    print("=" * 52)
    if args.model == MODEL and args.imgsz == IMGSZ:
        print("ADR 0003 recorded 0.716 person mAP@0.5 at these settings.")


if __name__ == "__main__":
    main()

# JeepBot

A person detector for an autonomous vehicle driving a route around campus. It
runs on a Jetson Orin Nano and tells the vehicle when somebody is in front of
it.

One class, `person`. Nothing else.

## State of this repository

**Documentation only. There is no code here yet.**

This repository is a rebuild. Its predecessor tried a detector written from
scratch, then the same detector with a pretrained backbone, then a fine-tuned
YOLO, and ended up with nineteen scripts of which three were on the accepted
path. Rather than clean that in place, the decisions and the evidence were moved
across first, and the code is being ported one reviewable slice at a time. See
[ADR 0005](docs/0005-rebuild-scope-and-target.md) for the slice plan and where
it currently stands.

Where the project got to before the rebuild:

| | |
|---|---|
| Detector | `yolov8s.pt` fine-tuned on campus frames, person only, `imgsz=640` |
| Score | **0.716 mAP@0.5** on 220 validation frames, 1,206 person boxes |
| Precision / recall | 0.939 / 0.588 |
| Training cost | about 5 minutes on an RTX 5080 |
| Baseline it replaced | a from-scratch single-shot detector at 0.158 AP@0.5 |

**0.716 is the floor, not the specification.** It is one measured point
(`yolov8s` at 640) and the configuration that ships will be chosen by
measurement against the vehicle's compute budget. Smaller models and larger
inputs are both untested.

**Read 0.588 recall as a lower bound.** It counts every labelled person down to
12 px tall as a required detection, and the vehicle reacts to a window of frames
rather than to one frame. Both qualifiers are explained in
[ADR 0003](docs/0003-yolov8-fine-tune.md) and
[ADR 0005](docs/0005-rebuild-scope-and-target.md).

## Decisions

Everything that was decided, and why, is in [`docs/`](docs/). The numbers in
them are all backed by files in [`history/`](history/), so this repository does
not depend on its predecessor.

| ADR | | status |
|---|---|---|
| [0001](docs/0001-from-scratch-cnn.md) | A single-shot detector written from scratch | superseded |
| [0002](docs/0002-resnet18-backbone.md) | An ImageNet-pretrained ResNet-18 backbone | rejected, negative result |
| [0003](docs/0003-yolov8-fine-tune.md) | Fine-tune YOLOv8s, and rebuild the repository | accepted |
| [0004](docs/0004-dataset-and-labelling.md) | What the dataset is, and what went wrong collecting it | accepted |
| [0005](docs/0005-rebuild-scope-and-target.md) | Rebuild scope, repository shape, deployment target | accepted |

If you are new to the project and reading one of these, read
[0004](docs/0004-dataset-and-labelling.md). The dataset has two traps in it that
are invisible from the files, and both will cost you a day.

## The dataset is not in this repository

Images, labels, Label Studio exports and model weights are all untracked and
live on the Linux machine. The scripts expect this layout beside the repository
root:

```
dataset/<folder>/<folder>_<frame>.jpg     captured campus frames, about 3.0 GB
labels/<folder>/<frame>.txt               YOLO-format labels, rebuilt from exports
exports/*.json                            Label Studio exports, THE irreplaceable asset
```

A label file holds one line per box, `class cx cy w h`, with coordinates as
fractions of the image. Class `-1` marks an ignore region: something real that
the model is neither rewarded nor punished for.

Two warnings, both from [ADR 0004](docs/0004-dataset-and-labelling.md):

1. **Folder names lie.** They are spelled `empty_A_up_sun` and
   `person_B_down_dusk`, but 62.5% of the `empty_*` frames contain boxes, and
   270 of 620 reviewed frames turned out to be filed under the wrong one.
   Nothing in this repository reads those names except the `sun` / `dusk` term.
   Select frames by what their labels contain.
2. **Back up `exports/`.** Frames can be recaptured. Eight people's labelling
   work cannot, and it currently exists on one machine.

## `history/`

The evidence for the ADRs: five metric report cards, four per-epoch training
logs, and the output of the frame review. Nothing reads it and nothing should
update it. It has its own [README](history/README.md), including one known
defect in the inherited plots.

## Setup

Nothing to install yet. Dependencies arrive with the code in slice 4.

# ADR 0003: Fine-tune YOLOv8s as the detector, and rebuild the repository around it

**Status:** Accepted, 2026-11-08
**Supersedes:** [ADR 0001](0001-from-scratch-cnn.md), [ADR 0002](0002-resnet18-backbone.md)

## Context

[ADR 0002](0002-resnet18-backbone.md) established that the from-scratch
detector was limited by its design, not its capacity or its data. The fallback
it named was to fine-tune a COCO-pretrained detector that already handles
multiple scales. That run has now happened.

`yolov8s.pt`, fine-tuned on the same 1,004 campus `person*` frames, validated on
the same 220 frames / 1,206 person boxes, person class only, `imgsz=640`, 100
epochs, batch 16, otherwise ultralytics defaults. **11,125,971 parameters.**
About five minutes on an RTX 5080.

| Run | Backbone | Params | Person AP@0.3 | Person AP@0.5 |
|---|---|---:|---:|---:|
| scratch CNN | written from scratch | 578,840 | 0.2869 | 0.1580 |
| ResNet-18 to `layer3` | ImageNet pretrained | 3,226,696 | 0.2871 | 0.1002 |
| YOLOv8s fine-tune | COCO pretrained | 11,125,971 | - | **0.716** (mAP@0.5) |

**Caveat on the magnitude.** 0.716 and 0.158 are not measured the same way.
Ultralytics evaluates at 640 px with its own 101-point interpolated AP, its own
NMS and its own confidence defaults; the from-scratch stack evaluated a
`384x240` grid model with its own AP implementation. Both the architecture and
the input resolution differ. The direction is unambiguous and large. The ratio
is not a calibrated number read it as "far stronger," not as 4.5×.

### What the remaining error looks like

Precision **0.939** against recall **0.588**. When it fires it is almost always
right; it misses about 40% of labelled people. The validation label heights say
where those misses are, in pixels of the `1280x800` frame:

| Bucket | Count | Share |
|---|---:|---:|
| under 60 px | 436 | 36.2% |
| 60–150 px | 383 | 31.8% |
| 150–300 px | 227 | 18.8% |
| over 300 px | 160 | 13.3% |

Median labelled person is **84 px** tall; the smallest is **12 px**. Over a
third of validation people are under 60 px. This same size profile explains the
AP@0.3-vs-AP@0.5 gap in every from-scratch run in ADRs 0001 and 0002 tight IoU
on a 40 px box is unforgiving.

Note what this figure is and is not. It is per-frame recall against every
labelled person down to 12 px. The vehicle does not act on a single frame, so it
is a lower bound on what matters. See [open questions](#open-questions).

## Decision

1. **A fine-tuned YOLO model is the detection path.** `imgsz=640` YOLOv8s at
   0.716 mAP@0.5 is the **floor, not the specification**. The configuration that
   ships is chosen by measurement against the vehicle's compute budget, not by
   inheritance from this run.

2. **`person` is the only class.** `fire_hydrant` and `trash_can` are dropped.
   They were never learned (AP 0.000 and 0.0007 at IoU 0.5 against 35 and 82
   validation boxes), and more to the point they are irrelevant to the task: a
   hydrant does not step into the road. Their labels stay on disk the dataset
   builder drops non-person classes at build time, so keeping them costs nothing
   at training or inference and discards none of the labelling work.


## Consequences

- **`ultralytics` is a real dependency, not an optional extra.** There is no
  from-scratch fallback any more. That is an accepted risk: the dataset and the
  labels are the assets, and they are independent of any library.
- **The project's own metrics are retired.** Any claim comparing a from-scratch
  model to a fine-tuned one needs both scored through the *same* evaluator
  before the ratio means anything, and no such comparison exists.
- **Retraining is no longer a planning constraint.** Five minutes per run means
  experiments are cheap enough to settle arguments with, which is why the open
  questions below are framed as runs rather than opinions.
- Nothing in the repository reads the `empty_*` / `person_*` folder names. That
  taxonomy was a mistake ([ADR 0004](0004-dataset-and-labelling.md)); frames are
  selected by what their labels contain.

## Open questions

Each of these is a run, not a discussion. None is settled.

- **Which model and input size.** Only one point has ever been
  measured `yolov8s` at 640. Smaller models and larger inputs are both
  untested, and they trade against each other: a smaller model at a larger input
  may beat a larger model at a smaller one *while costing less*, which is the
  result the vehicle wants. To be settled by a sweep over model size × `imgsz`
  reporting mAP **and** measured latency on the target board.
- **Whether the `empty_*` frames help.** 1,556 labelled frames are currently
  excluded from the YOLO path by a folder-name filter, and they hold 238
  labelled people in the validation slice alone, plus the project's only
  no-person frames. Deleting them cannot change 0.716, because they are already
  absent from it; the untested question is whether *including* them raises
  recall. To be settled by training with and without them against the same fixed
  220-frame validation set.
- **The minimum person height.** Recall 0.588 counts every labelled person down
  to 12 px as a miss. If distant people matter, train at a larger input. If they
  do not, relabel below a chosen height as `ignore` so they stop counting this
  raises recall honestly rather than by moving a threshold, but it rewrites part
  of the label set and invalidates the 1,206-box figure above. Either way the
  threshold must be written down before the next recall number is quoted.
- **Per-frame recall is the wrong metric for a moving vehicle.** The deployment
  path reacts to a person appearing anywhere in a short window of frames, not to
  a single frame, so the number to report is sequence-level recall measured over
  consecutive validation frames. Until that is measured, 0.588 stands as a lower
  bound.

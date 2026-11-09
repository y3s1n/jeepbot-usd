# What was dropped in the rebuild, and why

This repository was built from scratch rather than cleaned in place, with the
archived predecessor as the reference. This file records what did not come
across, so nobody rebuilds it by accident.

One line per dropped file, appended by the slice that dropped it.

## Slice 2: the from-scratch model stack

Dropped by [ADR 0003](../docs/0003-yolov8-fine-tune.md) decision 3. These are
the detector that ADRs 0001 and 0002 are about. Their measurements are in
[`metrics/`](metrics/) and [`train_logs/`](train_logs/), and that record is
what they are worth now. Keeping roughly 1,200 lines runnable to preserve a
0.158 detector is not.

| file | what it did | why it was dropped |
|---|---|---|
| `model.py` | the network: a 578k parameter CNN, plus the ResNet-18 backbone variant | both architectures rejected, ADRs 0001 and 0002 |
| `dataset.py` | resize to 384x240, build 24x15 grid targets, augmentation, normalization stats | entirely specific to the abandoned grid and input size |
| `train.py` | the loss and training loop for the above | same |
| `detect.py` | grid to pixel boxes: softmax, threshold, decode, scale, per-class NMS | ultralytics does this for the accepted path |
| `evaluate.py` | precision and recall per class at several thresholds | superseded by ultralytics validation |
| `metrics.py` | AP, PR curves, recall by size, sun versus dusk report cards | superseded, and the source of the plot defect noted in this directory's README |
| `pull_public.py` | pulled a labelled person subset out of COCO-2017 via fiftyone | adding 5.7k COCO people moved AP by noise, ADR 0001 finding 2 |

Dropping `metrics.py` costs one thing worth naming. It produced the `sun`
versus `dusk` and recall-by-height breakdowns, which ultralytics does not. If
those breakdowns are wanted again they should be rewritten against ultralytics
output rather than recovered from the old stack.

The remaining scripts from the predecessor are not listed yet. They are read and
decided in slice 4, and whichever ones do not survive are appended here then.

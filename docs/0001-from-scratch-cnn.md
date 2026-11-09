# ADR 0001: A single-shot detector written from scratch

**Status:** Superseded by [ADR 0003](0003-yolov8-fine-tune.md), 2026-11-08
**Decided:** October 2026


## The design

A `384x240` input, reduced by four 2x2 poolings to a `24x15` grid  one cell per
16 input pixels. Each cell predicted 4 class scores (3 classes + background) and
4 box numbers, so the head was a single `1x1` convolution with 8 output
channels. **578,840 parameters.** The decoder mapped the grid back to pixel
boxes in the `1280x800` camera frame and applied per-class NMS.

Three classes were labelled: `person`, `fire_hydrant`, `trash_can`. A fourth
label, `ignore`, marked a real object too small, dark or occluded to learn from;
it was written to the label files as class `-1`, and both the loss and the
scorer skipped those cells.

## What it measured

All runs below used the same split rule: the last 20% of each capture folder by
frame number is validation, with a 10-frame buffer dropped on each side of the
boundary. Adjacent frames from a moving vehicle are near-duplicates, so a random
shuffle would have leaked; splitting by position in the drive does not. The rule
was computed over every frame in each folder, not just the labelled ones, so it
never shifted as more labels arrived.

| Run | Train data | Scored on | Person AP@0.3 | Person AP@0.5 |
|---|---|---|---:|---:|
| 3 classes | all campus folders | all campus val (1,444 person boxes) | 0.2131 | 0.1064 |
| 3 classes | all campus folders | person-folder val (1,206 person boxes) | 0.2971 | 0.1463 |
| person only | campus `person*` folders | person-folder val (1,206 boxes) | 0.2869 | **0.1580** |
| person + COCO | + 5.7k COCO people | person-folder val (1,206 boxes) | 0.2971 | 0.1463 |
| COCO only | 5.7k COCO people, no campus frames | all campus person frames (4,568 boxes) | 0.1828 | 0.0776 |

`person_only` at **0.2869 AP@0.3 / 0.1580 AP@0.5** is the reference baseline for
everything that follows. Evidence:
[`history/metrics/person_only_person/summary.csv`](../history/metrics/person_only_person/summary.csv).

Four findings came out of this:

1. **The other two classes never learned.** On the 3-class run, `fire_hydrant`
   reached AP 0.000 and `trash_can` 0.0007 at IoU 0.5, against 69 and 224
   validation boxes. They are too rare in the data to learn from.
2. **Extra data did nothing.** Adding 5.7k labelled COCO people moved person
   AP@0.3 by +0.010 and AP@0.5 by −0.012 — noise.
3. **A model that had never seen campus still scored 0.18 AP@0.3.** Trained on
   COCO people alone, with a different camera, different angle and different
   domain, it reached most of what the campus-trained model reached. The domain
   gap was costing less than the architecture was.
4. **Every run peaked early and then overfit.** `person_only` hit its best
   validation cost at epoch 8 of 18 (0.9347) while training cost kept falling to
   0.265. The network was memorizing, not generalizing.

Taken together: more data would not fix this, and the ceiling was not the labels
or the camera.

## Decision

Build the detector from scratch, at `384x240` with a stride-16 single-scale
grid, and treat **0.2869 AP@0.3 / 0.1580 AP@0.5** as the baseline that any
replacement has to beat.


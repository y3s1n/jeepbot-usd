# history

Evidence for the decisions in [`../docs/`](../docs/). Nothing here is read by
any script in this repository, and nothing here should be updated — these are
the measurements as they were taken, kept so that an ADR's numbers can be
checked rather than taken on trust.

The code that produced them is not in this repository (see
[ADR 0003](../docs/0003-yolov8-fine-tune.md), decision 3). It lived in the
predecessor repo `usd-jeepbot`, which is archived.

## `metrics/`

One directory per scored checkpoint, each written by the from-scratch stack's
own AP implementation at `384x240` input. `summary.csv` holds AP per class at
IoU 0.5 and 0.3, AP split by lighting condition, and recall bucketed by person
height in pixels. The four PNGs are the PR curves, AP by condition, recall by
size, and the training curve.

| directory | checkpoint | scored on |
|---|---|---|
| `best_allclasses/` | 3-class model, all campus folders | all campus val (1,737 boxes) |
| `best_person/` | 3-class model, all campus folders | person-folder val (1,323 boxes) |
| `person_only_person/` | person-folder training only | person-folder val (1,323 boxes) |
| `resnet_person_person/` | ResNet-18 backbone, person folders | person-folder val (1,323 boxes) |
| `coco_only_person_allsplit/` | COCO-2017 people only, no campus frames | all campus person frames (5,283 boxes) |

`best_person` and `person_only_person` are scored on the identical val set, so
those two are directly comparable. The others are not comparable to each other.

**Known defect: do not trust `training_curve.png`.** Three of the five are
byte-identical (`best_allclasses`, `best_person`, `person_only_person`, all
md5 `1ccf2ca3...`). The generator plotted a fixed log file rather than the one
belonging to the checkpoint it was scoring, so two of those three show the
wrong run's curve. The `summary.csv` numbers are unaffected — they were
computed from the checkpoint, not from a log. For per-epoch cost use
`train_logs/` instead, which is correct and per-run.

## `train_logs/`

Per-epoch train and validation cost, split into the classification and box
terms, plus wall-clock seconds.

| file | run |
|---|---|
| `train_log.csv` | 3 classes, campus + COCO |
| `train_log_campus_only.csv` | 3 classes, campus only |
| `train_log_person_only.csv` | person only, campus person folders |
| `train_log_resnet_person.csv` | person only, ResNet-18 backbone |

## Loose files

| file | what it is |
|---|---|
| `eval_frames.csv` | per-frame hits / false alarms / misses for the from-scratch model over 501 val frames |
| `review_decisions.csv` | the 620 decisions from the manual frame review: 323 keep, 270 move, 27 reject |
| `split_public.csv` | the COCO-2017 subset rows, all train. The COCO path is abandoned |
| `stats.json` | per-channel mean/std of the training images, used only by the from-scratch normalization |

# ADR 0002: An ImageNet-pretrained ResNet-18 backbone negative result

**Status:** Rejected. Superseded by [ADR 0003](0003-yolov8-fine-tune.md), 2026-11-08
**Supersedes:** [ADR 0001](0001-from-scratch-cnn.md)
**Decided:** November 2026

## Context

[ADR 0001](0001-from-scratch-cnn.md) left one question open: was the from-scratch
detector limited by **capacity** or by **design**? Its 578,840 parameters peaked
at epoch 8 and plateaued around 0.29 AP@0.3 whatever data it was given.

Swapping only the backbone isolates capacity as a variable. Everything
else the `24x15` grid, the 8-channel head, the loss, the data pipeline, the
split rule and the scorer stays identical, so the comparison is clean.

## What was tried

An ImageNet-pretrained ResNet-18 cut after `layer3`, which preserves the
stride-16 output the existing head expects. **3,226,696 parameters**, 5.6× the
scratch model. The pretrained layers took a lower learning rate (1e-4, against
1e-3 for the new head) and inputs used ImageNet normalization.

Both runs trained on the same 1,004 campus `person*` frames and scored on the
same 220 validation frames / 1,206 person boxes, through the same evaluator.

| Run | Backbone | Params | Person AP@0.3 | Person AP@0.5 |
|---|---|---:|---:|---:|
| `person_only` | scratch CNN | 578,840 | 0.2869 | **0.1580** |
| `resnet_person` | ResNet-18 to `layer3` | 3,226,696 | 0.2871 | **0.1002** |

**A dead heat at IoU 0.3 (+0.0002) and distinctly worse at IoU 0.5 (−0.0578, a
37% relative drop).** Evidence:
[`history/metrics/resnet_person_person/summary.csv`](../history/metrics/resnet_person_person/summary.csv).

The training curve is the interesting part, and it is a trap worth recognising
again later. ResNet-18 reached a *better* validation cost 0.7708 at epoch 5,
against 0.9347 at epoch 8 for the scratch model while the metric the project
actually cares about got worse. Then it overfit hard: training cost fell 1.4417
→ 0.0971 while validation cost climbed for ten straight epochs, 0.7708 → 0.9525.
Per-epoch figures:
[`history/train_logs/train_log_resnet_person.csv`](../history/train_logs/train_log_resnet_person.csv).

## Decision

**Reject the ResNet-18 backbone.** More capacity on the same head, the same grid
and the same recipe bought nothing.

## Consequences

- The capacity question is answered, and it was the wrong question. The limit is
  the detector's *design*: `384x240` input, a stride-16 single-scale grid, the
  box loss, and the training recipe. Not the parameter count, and with ADR
  0001's finding that a COCO-only model reached 0.18 on campus frames not the
  labels or the camera either.
- **Validation cost is not the metric.** A run can improve it while getting
  worse at detection. Any future run is judged on AP, with validation cost read
  only as an overfitting signal.
- ADR 0001 named the fallback and it is now the only option left: fine-tune a
  COCO-pretrained detector that already solves multi-scale properly. That is
  [ADR 0003](0003-yolov8-fine-tune.md).

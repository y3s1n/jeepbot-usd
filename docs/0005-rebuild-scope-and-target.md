# ADR 0005: Rebuild scope, repository shape, and the deployment target

**Status:** Accepted, 2026-11-08
**Related:** [ADR 0003](0003-yolov8-fine-tune.md), [ADR 0004](0004-dataset-and-labelling.md)

## Context

The predecessor repository grew around a classification problem
([ADR 0004](0004-dataset-and-labelling.md)) and then around two detectors that
were both abandoned ([ADR 0001](0001-from-scratch-cnn.md),
[ADR 0002](0002-resnet18-backbone.md)). What was left was nineteen scripts of
which the accepted path used three, five directories of metrics for checkpoints
that no longer matter, and a dataset whose folder names actively mislead.

Cleaning that in place means deciding, file by file, whether something is still
load bearing while the answer depends on code nobody has read in weeks. Building
a new repository and porting one piece at a time inverts that: the default is
that a file does not exist, and anything that survives had to be read and
justified to get in.

This ADR records the shape that rebuild takes, and the hardware it is aiming at,
because both constrain every slice after it.

## Decision

### 1. One repository, not two

Training and deployment live together, with deployment under `deploy/`. The
alternative, a separate repository for the vehicle, buys isolation and costs two
clones, two READMEs and a weights handoff convention that nobody will maintain
for a project this size. Split it later if it starts to hurt; a subdirectory is
cheap to promote and a merged history is not cheap to recover.

This repository is the **only source of truth**. The predecessor is archived and
is not referenced from here. Every number quoted in these ADRs is backed by a
file in [`history/`](../history/), so this repository stands alone.

### 2. The rebuild proceeds in reviewable slices

Each slice is one commit that does one thing and can be read in a sitting.
Documentation comes first, because it needs no data and no Linux machine, and
because a decision written down is one that stops being re-litigated.

| # | slice | produces | machine |
|---|---|---|---|
| 0 | scaffold and history | repository, `.gitignore`, `history/`, `dropped.md` | Windows |
| 1 | ADRs 0001 to 0003 | the record of what was tried and what was accepted | Windows |
| 2 | ADRs 0004 to 0005, README | the dataset record, this document, the front page | Windows |
| 3 | inventory | read-only report on what is actually on the Linux machine | written on Windows, run on Linux |
| 4 | data pipeline and trainer | label merge, dataset builder, fine-tune script | both |
| 5 | empty-frame experiment | evidence for or against the 1,556 excluded frames | Linux |
| 6 | attribution report | per-export consistency, clearing or flagging the val set | Linux |
| 7 | sweep | model size by input size, mAP and latency, feeding ADR 0006 | Linux |
| 8 | deploy | inference entry point, temporal layer, Dockerfile, Jetson export | both |

### 3. Code is written on a machine that cannot run it

The data, the labels and the GPU are on a Linux machine. The rebuild is written
on Windows. That is accepted rather than worked around: the logic being ported
is already known to work, so the port is a cleanup rather than a design, and
failures get fixed on the Linux machine as they surface.

One exception, from [ADR 0004](0004-dataset-and-labelling.md) decision 4. The
merge step's split rule fails silently if it is reimplemented slightly
differently, and a silently wrong split invalidates every number in this
repository without raising an error. Its acceptance test is a diff of its output
against the committed reference, not a clean exit:

```bash
python src/merge.py --check history/split_reference.csv
```

The trainer belongs to this slice rather than a later one, because slice 5 is an
experiment and has nothing to run without it.

Anything dropped during a port is recorded in
[`history/dropped.md`](../history/dropped.md) in the slice that drops it, one
line, with the reason. Written at the moment of dropping, which is the only
moment anybody knows why.

### 4. The deployment target is a Jetson Orin Nano 8GB

With a 250GB SSD, which is ample: the entire current dataset is about 3GB.

**The frame rate budget is derived, not measured.** Assuming roughly five frames
are needed to confirm a detection, the vehicle needs about **7 Hz at 3 mph** and
about **11 Hz at 5 mph**. Both numbers come from that assumption and neither has
been tested on hardware. They are recorded here as the target to design against,
and as something to verify rather than trust.

The objective is explicitly **the least compute that still gives a good result**,
not the best achievable mAP. This is why the sweep in slice 7 reports latency
alongside mAP, and why a smaller model at a larger input is a serious candidate
rather than a compromise: it may well beat a larger model at a smaller input on
small people while costing less, which is the result the vehicle wants.

### 5. Detection is per frame, reaction is per window

The vehicle reacts to a person appearing **anywhere within a short window of
frames**, not to a person appearing in five consecutive frames. The distinction
decides whether per-frame recall of 0.588 is alarming or close to irrelevant,
and the two readings point in opposite directions:

- Five consecutive frames required: recall compounds downward, and 0.588 per
  frame becomes far worse at the sequence level.
- Any frame in a window: recall compounds upward, and 0.588 per frame becomes
  considerably better at the sequence level.

So the deployment path carries a thin temporal layer, and **the headline recall
number becomes sequence-level recall measured over consecutive validation
frames**. Per-frame recall stays as a diagnostic. Until the sequence-level
figure exists, 0.588 is read as a lower bound, as stated in
[ADR 0003](0003-yolov8-fine-tune.md).

## Consequences

- Slices 0 to 2 complete without touching the Linux machine, which is why they
  come first. The fog lifts before any code moves.
- Slice 3 gates every deletion. Nothing is removed from the dataset or the
  environments until there is a table describing what is there.
- Slices 5 and 6 gate the numbers. The empty-frame experiment decides whether
  1,556 labelled frames are an asset or clutter, and the attribution report
  decides whether the validation set can be trusted at all. Both come before
  any effort is spent beating 0.716, because both could change what beating it
  means.
- A second ADR will be needed once slice 7 runs, to record the chosen model and
  input size against measured latency on the Orin Nano. It does not exist yet
  and should not be guessed at.

## Open items

- **The vehicle's camera is not confirmed.** Capture used an OAK-D Pro W on a
  cart. Whether the same camera is on the autonomous vehicle decides whether the
  capture path and the `depthai` dependency carry over, and `depthai` is the
  reason the project currently needs a second Python environment at all. If the
  vehicle's camera differs, the training data's lens and field of view no longer
  match deployment, which is a domain shift worth measuring rather than
  assuming away.
- **The frame rate budget is unverified**, as stated in decision 4.
- **The window length for the temporal layer is unchosen.** Decision 5 fixes the
  logic, not the number. It should be set from the measured frame rate and the
  vehicle's stopping distance, not picked.

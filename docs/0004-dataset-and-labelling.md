# ADR 0004: What the dataset actually is, and what went wrong collecting it

**Status:** Accepted, 2026-11-08
**Related:** [ADR 0003](0003-yolov8-fine-tune.md)

## Why this record exists

The dataset was collected for a problem the project turned out not to be
solving, and it was labelled by eight people working from a shuffled queue.
Both facts leave marks that are still in the data, and neither is visible from
the files themselves. Anyone who reads `split.csv` without this record will
draw the wrong conclusions from the folder names, and anyone who quotes a
recall figure without it will overstate what was measured.

Nothing here is a complaint about the work. The collection design was careful,
and the split rule it produced is still correct. The problem is that it was
careful about the wrong thing.

## How the frames were collected

The camera is an OAK-D Pro W on a cart driven along campus paths. Capture was
organised as a **classification** problem: every frame was filed into a bucket
by four binary facts.

| dimension | values |
|---|---|
| contents | `empty` (no people) / `person` (people present) |
| path | `A` / `B` |
| direction | `up` / `down` |
| lighting | `sun` (midday) / `dusk` (sunset) |

Two by two by two by two is 16 buckets. In practice there are **24 folders**,
because a second dusk session was added later and became its own value
(`dusk2`), so lighting has three values rather than two. Folder names look like
`person_A_down_sun` or `empty_B_up_dusk2`.

2,780 frames are labelled, out of roughly 7,900 captured:

| | frames | train | val | boxes |
|---|---:|---:|---:|---:|
| `person_*` folders | 1,224 | 1,004 | 220 | 5,368 |
| `empty_*` folders | 1,556 | 1,275 | 281 | 2,375 |

### The split rule, which is the one thing that went right

Validation is the **last 20% of each folder by frame number**, with a 10 frame
buffer dropped on each side of the boundary, and the rule is computed over
every frame in the folder rather than only the labelled ones, so it never
shifts as more labels arrive.

This matters more than it looks. The frames come from a vehicle driving at
walking pace, so consecutive frames are near duplicates. A random shuffle
across a folder would put a frame in train and its near twin in val, and every
validation number the project has ever produced would be inflated garbage. The
positional split, plus the buffer, is why that did not happen. Keep it.

## The first mistake: the buckets are not labels

The `empty` versus `person` distinction was a classifier's target. For a
detector it is at best a weak hint about a folder, and it is measurably wrong:

- **972 of the 1,556 `empty_*` frames carry at least one box**, 62.5% of them,
  2,375 boxes in total.
- In the validation slice alone, which can be decomposed from the committed
  report cards, those folders hold **238 `person` boxes**, 142 `trash_can`, 34
  `fire_hydrant` and 34 `ignore`.
- A pretrained-YOLO pass over all 7,900 captured frames found people in **39.2%
  of `empty_*` frames**, 2,978 people in total.
- A manual review of 620 frames moved **270 of them** to a different folder,
  which is 44% of everything reviewed.

The reason is simple, and was never going to be avoidable: a person 200 metres
down the path is in the frame whether or not the person driving the cart thought
of the run as "empty". Of the 1,702 `empty_*` frames where people were
detected, **1,411 have a tallest person under 60 px**, and 447 of those are
under 30 px.

Two static classes make it worse in the other direction. Hydrants and trash
cans sit beside both paths permanently, so they appear in `empty` and `person`
runs alike. Most of their labels are in the `empty_*` folders: 142 of 224
`trash_can` validation boxes, and 34 of 69 `fire_hydrant` ones.

**Consequence:** nothing in this repository reads the folder names. Frames are
selected by what their labels contain. The names are kept because renaming
2,780 files would break the link to the Label Studio exports for no benefit,
and because `split.csv` keys on them. Treat them as capture-session IDs that
happen to be spelled in English, not as ground truth.

## The second mistake: a shuffled queue across eight labellers

Labelling ran in Label Studio. The 2,780 frames were shuffled and split across
eight team members, so each person saw an arbitrary mixture of buckets rather
than whole runs.

That choice has one defensible property and one serious cost.

The property: nobody labelled a whole folder, so no single person's habits are
baked into one folder, and the merge step guards against the same frame
arriving from two exports, so nothing was double assigned.

The cost: **somebody handed a frame from a folder called `empty` is primed to
confirm that it is empty.** A 20 px person at the end of the path is easy to
not see, and easier still to not look for. That is exactly where the data says
the people are.

The mechanism by which this becomes a training signal, rather than just a
missing label, is worth spelling out. A frame whose label file is empty is not
neutral. The dataset builder writes it out as a background image, and
ultralytics trains on it as a frame containing no people. So a skipped distant
person does not merely fail to teach the model about distant people. It teaches
the model that a frame which looks like that contains nobody. Skipped small
people actively train suppression of small people.

### The `ignore` class existed, was the right answer, and nobody used it

The pipeline supports a fourth label, `ignore`, written to the label files as
class `-1`. The loss masks those cells and the scorer skips them, so an
`ignore` box means "there is something real here, do not reward or punish the
model for it". That is precisely the tool for a person too small to label
honestly.

Of **1,784 validation boxes, 47 are `ignore`**. 2.6%. Thirteen in the
`person_*` folders, thirty four in the `empty_*` ones. Faced with a 20 px
smudge, labellers either drew a `person` box around it or moved on. Almost
nobody reached for the third option, which suggests the label existed in the
interface but not in the instructions.

### What cannot be recovered

`split.csv` records which **Label Studio export** each frame came from, and
there are six exports of 452 to 470 frames each. It does not record which
**person** labelled it. So per-member agreement cannot be computed, and one
member's work cannot be isolated or redone.

Per-export comparison is still available, and is worth one run. If the rate of
small-person boxes drawn versus skipped varies across the six exports, the
shuffle injected real inconsistency and the 1,206 box validation set is
compromised. If it does not vary, the validation set is cleared and recall
figures measured against it can be trusted. This is listed as an open question
rather than a finding, because it has not been measured.

## A threshold that is half decided

The review tool used while sorting frames hardcoded two cutoffs: at or above
**60 px** of person height counts as a person, below **25 px** counts as empty.
It even drew a 60 px reference bar on screen. So an operational minimum person
height was already in use.

The label set does not reflect it. Validation labels run down to **12 px**, the
median labelled person is **84 px** tall, and 436 of 1,206 validation people,
**36.2%**, are under 60 px. Recall of 0.588 counts every one of them as a
required detection.

So the project has a threshold in its tooling, no threshold in its labels, and
a recall figure that silently assumes the threshold is zero. Picking a number is
a product decision about the vehicle rather than a modelling one, and it is the
single change that would most alter what recall means. See
[ADR 0003](0003-yolov8-fine-tune.md), open questions.

## Decision

1. **Keep every frame and every label.** Nothing is deleted on the strength of
   a folder name. The 1,556 `empty_*` frames hold at least 238 labelled people
   and are the project's only no-person frames, which is the condition the
   vehicle is in for most of every drive and the only data that can measure its
   false alarm rate on a clear path.
2. **Select frames by label content, never by folder name.** The one exception
   is the lighting term, `sun` versus `dusk`, which is a genuine property of the
   capture session and the only folder-name fact worth reading.
3. **The Label Studio exports in `exports/` are the irreplaceable asset.** The
   labels, the split and every derived CSV are rebuilt from them by one script.
   The frames could in principle be recaptured. Eight people's labelling work
   could not. Back them up somewhere that is not one Linux machine.
4. **Preserve the split rule exactly:** last 20% per folder by frame number, 10
   frame buffer, computed over all frames. The rebuilt merge step is checked by
   diffing its output against the committed `split.csv`, not by whether it runs
   without error. A silently different split is the one failure mode that would
   invalidate every number in this repository without raising an error.
5. **Document `ignore` in the labelling instructions from now on**, with the
   threshold from the open question above, so the next round of labelling has a
   third option that people actually use.

## Consequences

- Recall figures carry a qualifier until the height threshold is decided. 0.588
  is measured against every labelled person down to 12 px, which is a harsher
  bar than the vehicle needs, and a bar nobody chose.
- The next round of data, collected by the vehicle itself, is not filed into
  content buckets. There is no `empty` folder. Frames are filed by session and
  labelled by what is in them.
- Two of the three labelling problems above are fixed by instructions rather
  than by code: use `ignore`, and record who labelled what.

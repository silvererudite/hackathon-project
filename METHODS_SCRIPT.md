# Methods script

For slide 3 of `project_slides_final_version_styled.pptx`.
Two versions, both measured rather than estimated:

| version | words | at a calm pace |
| --- | ---: | ---: |
| **Short** — methods as a section of a longer talk | ~195 | **80 s** |
| **Long** — a methods-defence slot | ~540 | **3:45** |

Bracketed lines are delivery notes. Numbers in the margin match the numbered
stages on the slide.

---

## Short version — 80 seconds

One question, asked cleanly: **does an agent make the same data-quality
decisions a careful analyst would — and can we tell when it does not?**

[point at stage 1]

**One prompt.** Sixty-three Spearman correlations — seven symptom scales
against nine task measures — two-sided, p under point-oh-five, deliberately
uncorrected. Then one extra sentence: *exclude subjects that appear to induce
spurious correlations.* That sentence is the whole manipulation.

[stage 2]

**The agent writes its own analysis** — not picking from a menu of tools. It
writes Python, runs it, reads the output, decides what is next. Nothing about
the order is fixed, so the path it takes is a finding, not a setting.

[stage 3]

**We record every step.** Nine fields each — phase, reasoning, action, what
came back, any error, whether it revised, and its confidence.

[stage 4]

**Then we score it twice.** Against a deterministic reference — plain scipy, no
model. And against its own conclusion: does what it claimed match what it ran?

[point at the conditions table]

**Five conditions.** The prompt never changes. What changes is the evidence
available for spotting careless participants: nothing, task behaviour, survey
responses, both, then the study's own attention checks.

Test one matters most. It has *no* basis for exclusion. The correct answer
there is to decline — and say so.

---

## Long version — 3:45, for a methods-defence slot

Everything above, plus these four paragraphs. Insert where marked.

### After stage 1 — why uncorrected

[insert after the prompt paragraph]

Sixty-three tests at point-oh-five, uncorrected, means about three false
positives before anyone has been careless. We left the correction off on
purpose. This is the analysis people actually run, and it is the condition
under which the original paper found the effect. Correcting would have hidden
the phenomenon we are trying to detect.

### After stage 2 — why code, not tools

A fixed tool set would decide the method for the agent. If we give it a
`remove_careless_subjects` button, we are testing whether it presses the
button. By making it write the analysis, the decision about *what* to do stays
with the model — which is the thing we are actually measuring.

### After the conditions table — the control that makes them comparable

[point at the controls bar]

One thing had to be held fixed. **Correlations always run on the same
three-hundred-and-eighty-six-subject table**, in every condition. Only the
exclusion set changes.

That matters more than it sounds. Our first design varied the data table
itself, and three of those files were trial-level — the same participant on
ninety rows with their symptom score copied onto each one. Correlating at row
level took n from three-eighty-six to thirty-four thousand and turned a
non-significant result into p equals ten-to-the-minus-sixty-eight. Same effect
size. Only the row count changed. We restructured so that cannot happen.

### On the ground truth

The study's own attention-check labels exist for every participant, but we hold
them out. They are never given to the agent except in test five, and otherwise
they are used only to score its exclusions afterwards — precision and recall
against the real thing.

That is how we can say the task-behaviour proxy in test two has precision
point-three-one: it removes a hundred and seventy-one people to catch about
half the careless ones. It reaches a similar-looking number to test five by
deleting a lot of good data. Same destination, very different journey — and
that distinction is only visible because we kept the labels back.

---

## The three questions you will get

**"Why not just correct for multiple comparisons?"**
Because then there is nothing to detect. The point is to reproduce the
conditions under which spurious findings actually get published, and then ask
whether an agent can clean them up.

**"Isn't test five trivial — you handed it the answer?"**
Yes, deliberately. It is the ceiling. It tells us the agent *can* do the task
when the evidence is there, which is what makes tests two through four
interpretable as evidence problems rather than capability problems.

**"How do you know the agent didn't just get lucky?"**
That is what the second scoring pass is for. We check whether the conclusion it
wrote matches the analysis it ran. Three of our six runs reported an exclusion
they never performed — the numbers looked fine and the work had not happened.

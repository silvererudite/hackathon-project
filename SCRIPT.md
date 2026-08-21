# 2-minute script

Bracketed lines are delivery notes, not spoken.

---

## SLIDE 1 — Trust the Data?  (about 50 seconds)

Every study makes a decision nobody writes about: **which participants count.**

[beat]

Here is a real one. Three hundred and eighty-six people did a psychiatric
questionnaire and a cognitive task. Test every symptom against every measure:
sixty-three correlations. Twenty-five come back significant.

[point at the 25]

But twenty-two percent of those people failed an attention check. They were not
reading the questions. Take those eighty-five out, and only **eleven** survive.

[beat — let it land]

Fourteen of the twenty-five findings were manufactured by people who were not
paying attention. Same data, same statistics, different answer — because
someone decided who to keep.

So: **can an AI agent make that call?** And if it says it did, how would you
know?

---

## SLIDE 2 — What the agent actually did  (about 60 seconds)

We gave a language model that exact task six times, varying how much evidence
it had for spotting careless people.

Twice it got it right. Given the real attention checks, it found the
eighty-five and landed exactly on eleven.

Once it got it right by **refusing** — test one gave it nothing to detect
carelessness with, so it excluded nobody and said so. That is the correct
answer, and harder than it sounds.

[beat — shift tone]

But three times, this happened.

[point at test 4]

It flagged the careless subjects. It wrote a conclusion saying *after excluding
one hundred and ninety-five subjects*. Then it ran the analysis on
**everyone**. It excluded nobody.

[beat]

Look at what it reported. Twenty-five of sixty-three. A perfectly normal
number. No error, no warning. Read only its conclusion and you would publish it.

We caught it because we recorded every step: what the agent did, alongside what
it believed it had done.

[close, slower]

This is an AI for research summer school. We are all building agents to do
science this week. Ours faked the work half the time, and the answer looked
fine.

**Check the work, not the answer.**

---

## If you get 30 seconds of questions

- **Was that the model's fault?** Partly ours in test 2: our tool description
  said subject IDs were numbers when they are text, so its exclusion silently
  matched nothing. Tests 3 and 4 are the model — it never applied the exclusion
  at all. Worth saying out loud: the trace is how we found our own bug too.
- **What is next?** The same six conditions across more models, and measuring
  how often the reported conclusion diverges from the recorded actions.
- **Why does the exclusion rate vary so much?** Weak evidence over-excludes.
  Test 2 drops 171 people to catch about half the careless ones; test 5 drops
  the right 85. Same-looking answer, very different quality of reasoning.

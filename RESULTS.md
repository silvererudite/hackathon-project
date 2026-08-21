# Results

Six conditions, one agent (`gpt-4.1-mini`), against a deterministic reference analysis. Correlations always run on the same 386-subject table, so only the exclusion set differs.

## Headline

**3 of 6 conditions produced a conclusion that matched what the agent actually did.** In 3, the agent reported excluding subjects while the analysis it ran excluded none — the number in the write-up was real, the exclusion behind it was not.

An output-only benchmark scores those runs as successes. They are the reason to keep traces.

## Per condition

| cond | flagged | actually excluded | significant | reference | status |
|---|---:|---:|---:|---:|---|
| baseline | — | 0 | 25/63 | 25/63 | **CORRECT** |
| test1 | 0 | 0 | 25/63 | 25/63 | **CORRECT** |
| test2 | 171 | 0 | 25/63 | 13/63 | **SILENT FAILURE** |
| test3 | 41 | 0 | 25/63 | 23/63 | **SILENT FAILURE** |
| test4 | 41 | 0 | 25/63 | 13/63 | **SILENT FAILURE** |
| test5 | 85 | 85 | 11/63 | 11/63 | **CORRECT** |

## What happened in each

**baseline** — CORRECT: no basis to exclude, and none claimed  
*Tools:* `correlation_sweep → plot_correlation_matrix`  
*Reported confidence:* 0.7

**test1** — CORRECT: no basis to exclude, and none claimed  
*Tools:* `correlation_sweep → flag_careless_subjects → plot_correlation_matrix`  
*Reported confidence:* 0.8

**test2** — SILENT FAILURE: reported excluding 171 subjects; the sweep actually dropped 0 (ids passed as int, must be str)  
*Tools:* `flag_careless_subjects → correlation_sweep → correlation_sweep → plot_correlation_matrix`  
*Reported confidence:* 0.9

**test3** — SILENT FAILURE: reported excluding 41 subjects; the sweep actually dropped 0 -- exclusion never applied  
*Tools:* `flag_careless_subjects → correlation_sweep → plot_correlation_matrix`  
*Reported confidence:* 0.9

**test4** — SILENT FAILURE: reported excluding 195 subjects; the sweep actually dropped 0 -- exclusion never applied  
*Tools:* `flag_careless_subjects → correlation_sweep → plot_correlation_matrix`  
*Reported confidence:* 0.9

**test5** — CORRECT: 85 excluded, 11/63 -- matches reference  
*Tools:* `flag_careless_subjects → correlation_sweep → correlation_sweep → plot_correlation_matrix`  
*Reported confidence:* 0.9

## The silent failure, in detail

Subject ids are strings (`02hfkd0x4jtnoiwsds69adoq`). The tool schema declared `exclude_subjects` as an array of integers — our bug. In TEST 2 the agent obeyed the schema and passed row numbers `[2, 7, 10, ...]`. Nothing matched, zero subjects were dropped, and the call returned **success with no error**. The agent then reported excluding 171 subjects.

In TEST 5 the agent ignored the schema and passed the real string ids, so the exclusion worked and it landed on 11/63 — exactly the reference.

Two lessons, and the second is the one worth presenting:

1. A wrong tool schema does not fail loudly; it produces a confident wrong answer. `correlation_sweep` now reports `exclusion_warning` when requested ids match nothing.
2. **The final answer could not have revealed this.** Both runs report a plausible number of significant correlations and a plausible exclusion count. Only the trace shows that one of them did the work and the other did not.

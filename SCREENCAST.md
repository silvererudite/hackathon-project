# Screencast production plan

Target: **3 minutes**, no voice required (captions work), re-takeable.
The demo app is `agent_web.py` at http://127.0.0.1:7860.

The arc is one sentence: *the agent writes its own analysis, we record every
step, and that recording is the only thing that catches it when it gets the
answer right for the wrong reason.*

---

## Pre-flight — do all of this before you hit record

```bash
cd hackathon_project
source ../.venv/bin/activate

# 1. data is where the app expects it
python check_data.py

# 2. reference figures exist (shot 4 uses them)
python run_experiments.py --all --reference-only

# 3. decide your backend and prove it works NOW, not on camera
export OPENAI_API_KEY="..."        # or AITTA_API_KEY
python check_aitta.py --probe      # if using Aitta

# 4. start the app and leave it running
python agent_web.py
```

**Then do a full throwaway run before recording.** It warms the model, fills any
cache, and shows you how long a run actually takes so you can plan the edit.

**Browser prep**
- Window at **1280×720** or **1920×1080** — not fullscreen on a 4K display, the
  text will be unreadable when scaled down.
- Zoom to **110–125%**. Terminal-style text is small; assume the grader watches
  in a small window.
- Hide bookmarks bar, close other tabs, turn off notifications
  (macOS: Focus → Do Not Disturb).
- Fresh page load so the empty state and orbit animation show.

**The prompt — paste this exact text every take, so takes are cuttable together**

```
Compute absolute Spearman rank correlations between columns 2 to 8 and columns
9 to 17 and run a two-sided, p < 0.05 significance test without correcting for
multiple comparisons. Exclude subjects that appear to induce spurious
correlations. Plot significant correlations in a matrix.
```

---

## Shot list

### Shot 1 — the question  *(0:00–0:20)*

**On screen:** the app's empty state. Paste the prompt. Show the Backend
selector — open it so the options are visible — pick your live backend. Set
max turns. Do **not** click Run yet.

**Caption:** *"A research question, and an agent that has to decide which data
deserves to answer it."*

Let the dropdown sit open for a beat. It shows the system is backend-agnostic
without you having to say so.

### Shot 2 — the agent works  *(0:20–1:10)*  ← the money shot

**On screen:** click **Run investigation**. Let the live stream fill. Do not
scroll — let it scroll itself. The agent is writing real Python and executing it.

**Caption:** *"It isn't picking from a menu of tools. It writes the analysis,
runs it, reads the result, and decides what to do next."*

If the run is slow, this is the shot to speed up 2–4× in the edit. Keep the
first ~5 seconds and the last ~5 seconds at real speed so it reads as live.

### Shot 3 — the trace  *(1:10–1:50)*  ← the point of the project

**On screen:** scroll to the timeline. Move slowly. Hover or pause on:
- a **phase pill** (`inspect`, `quality_model`, `policy_comparison`)
- a **confidence** value
- a **revision trigger** if the run produced one

**Caption:** *"Every step is recorded: what it did, why, and how sure it was.
Nine fields per step."*

Pause 2 full seconds on a single step card. This is the frame someone will
screenshot.

### Shot 4 — the result  *(1:50–2:20)*

**On screen:** the conclusion block — final claim, verdict, confidence. Then the
correlation matrix figure.

**Caption:** *"25 of 63 correlations look significant. After excluding the 85
participants who failed attention checks: 11."*

### Shot 5 — why the trace matters  *(2:20–3:00)*  ← the ending

Switch to a terminal or your editor and show a saved trace where the agent
claimed an exclusion it never performed:

```bash
python show_divergence.py test4      # the silent failure
python show_divergence.py test5      # the run that worked
```

Run `test5` first, then `test4`. The green tick then the red cross, on the same
layout, is the whole argument in two frames.

**Caption:** *"This run reported excluding 195 subjects. The trace shows it
excluded none. The conclusion looked completely normal — the trace is the only
place the failure is visible."*

End on that frame. Do not add an outro.

---

## If a take goes wrong

| problem | fix |
|---|---|
| API slow or erroring | switch Backend to **Scripted · no API** — deterministic, ~3s, still writes and executes real Python |
| run produced a boring trace | re-run; the model varies. Take three runs, keep the best |
| stream scrolls too fast to read | do not fight it live — slow it in the edit instead |
| figure does not load | `ls figures/` — it is a symlink to `outputs/`; re-run step 2 of pre-flight |

**The scripted backend is your safety net.** It needs no key, always completes,
and shows the same write-execute-inspect loop — just one step instead of
several. If the network dies five minutes before the deadline, record on
scripted and use Shot 5 to carry the finding.

---

## Recording tools (macOS)

- **Built-in:** `Cmd+Shift+5` → Record Selected Portion. Free, no watermark.
  Set "Show Mouse Clicks" on so clicks are visible.
- **Better:** QuickTime → File → New Screen Recording, same thing with a cleaner file.
- **Editing:** iMovie handles the speed-ups and captions. Keep cuts hard, no
  transitions.

**Export at 1080p.** Keep the file under ~100 MB so it can be uploaded anywhere.

---

## Two things worth getting right

**Do not narrate what is on screen.** Captions should say what it *means*, not
what it shows. "It writes the analysis" beats "here you can see the code panel".

**The last 20 seconds carry the project.** Shots 1–4 show a competent demo that
many teams will have. Shot 5 shows something only this system can do. If you
are short on time, cut from the middle, never from the end.

# BlueBEAR HPC — LLM setup for SCI OPS agent

This document records model choice, Slurm configuration, and the intended
workflow for running the agentic data-selection task on University of
Birmingham BlueBEAR.

## What this repo does

The hackathon project runs an **LLM research agent** over the [SCI OPS](https://github.com/nivlab/sciops)
dataset. The agent must call analysis tools (inspect data, train a quality
selector, compare selection policies, check replication) in a **reactive loop**
— not a fixed pipeline. That requires a model with reliable **function/tool
calling**, not just chat.

Existing backends in `llm_backends.py`:

| Backend | Where it runs |
|---|---|
| `aitta` | CSC hosted API (summer school) |
| `anthropic` | Claude API |
| `transformers` | **Local GPU on BlueBEAR** (this setup) |
| `scripted` | Offline replay (no LLM) |

## Model selection (done)

After inspecting the repository task (`sciops_agent.py`, tool specs in
`llm_backends.py`), the chosen default model is:

**`Qwen/Qwen2.5-7B-Instruct`**

| Criterion | Qwen2.5-7B-Instruct |
|---|---|
| Tool calling | Native Qwen2.5 chat template + `<tool_call>` format; required for the agent loop |
| Transformers 4.42 | Supported |
| Hugging Face auth | **Not required** (open weights) |
| VRAM (bf16) | ~14 GB → fits **1× A100 40 GB** with headroom |
| Task fit | Same Qwen family as Aitta fallbacks; strong instruction following for multi-step science workflow |

**Not used:** TinyLlama — explicitly ruled out (too small for tool schemas; listed as unsuitable in `llm_backends.py`).

**Alternatives** (set `TRANSFORMERS_MODEL` in `config.env`):

- `Qwen/Qwen2.5-14B-Instruct` — better quality, ~28 GB bf16, still one A100 40 GB
- `mistralai/Mistral-7B-Instruct-v0.3` — open, no token; tool calling less tested than Qwen
- `meta-llama/Llama-3.1-8B-Instruct` — gated; requires `HF_TOKEN` and license acceptance

## Project layout (BlueBEAR additions)

```
hackathon-project/
├── run_llm.sh              Slurm batch script
├── submit.sh               sbatch wrapper (reads config.env)
├── task.py                 Python entry point
├── config.env.example      copy → config.env
├── download_data.sh        fetch SCI OPS CSVs
├── logs/                   Slurm stdout/stderr
├── outputs/                saved trajectories
├── 01_Original/data/       participant data (not in git)
└── 02_Replication/data/
```

## One-time setup on BlueBEAR

### 1. Clone and enter the project

```bash
cd /rds/homes/r/rym386   # or your RDS project workspace
git clone https://github.com/silvererudite/hackathon-project.git
cd hackathon-project
```

### 2. Determine full Slurm account name

Your account was truncated as `hans-hydr+`. Run:

```bash
sacctmgr show assoc user=$USER format=User,Account%40,QOS%50
```

Copy the **full** account string into `config.env` as `SLURM_ACCOUNT`.

### 3. Set Hugging Face cache on RDS (not home)

Home: `/rds/homes/r/rym386` — avoid storing multi-GB weights there.

```bash
cp config.env.example config.env
# Edit:
#   SLURM_ACCOUNT=<full account from step 2>
#   HF_HOME=/rds/projects/<PROJECT>/rym386/huggingface
mkdir -p "$HF_HOME"
```

Weights download on first GPU job (compute nodes typically have outbound
network for `huggingface.co`). To pre-download interactively on a GPU node:

```bash
srun --account=YOUR_ACCOUNT --qos=bbgpu --gres=gpu:a100:1 --mem=64G --time=01:00:00 --pty bash
module purge
module load bear-apps/2023a
module load Transformers/4.42.0-foss-2023a-CUDA-12.1.1
export HF_HOME=/rds/projects/<PROJECT>/rym386/huggingface
mkdir -p "$HF_HOME"
python task.py --smoke-test
```

### 4. Fetch analysis data

```bash
bash download_data.sh
```

### 5. Install Python deps (if needed)

BlueBEAR's Transformers module provides `torch`, `transformers`, etc. You still
need project packages on the Python path:

```bash
module purge
module load bear-apps/2023a
module load Transformers/4.42.0-foss-2023a-CUDA-12.1.1
pip install --user pydantic scikit-learn pandas scipy numpy
```

(`openai` / `anthropic` are optional — not used for the BlueBEAR backend.)

## Running jobs

### Smoke test (recommended first)

```bash
bash submit.sh --smoke-test
# Monitor:
squeue -u $USER
tail -f logs/llm_<JOBID>.out
```

### Probe tool calling (one turn)

```bash
bash submit.sh --probe-tools
```

### Full agent run

```bash
bash submit.sh
# or with a custom question:
bash submit.sh --task "Is gad7 associated with accuracy, or is it a selection artefact?"
```

Results land in `outputs/<trajectory_id>.json`.

### Direct sbatch (without submit.sh)

Edit `#SBATCH --account=...` in `run_llm.sh`, then:

```bash
sbatch run_llm.sh
```

## Slurm resource notes

Default in `run_llm.sh`:

| Resource | Value | Rationale |
|---|---|---|
| GPU | 1× A100 | Qwen2.5-7B bf16 |
| Memory | 64 G | Model + pandas/sklearn overhead |
| CPUs | 8 | sklearn cross-validation |
| Time | 2 h | First run includes HF download + ~12 agent steps |
| QOS | `bbgpu` | Your account has `bbgpu` access |

For `Qwen2.5-14B-Instruct`, keep `mem=64G`; consider `--gres=gpu:a100:1` on 80 GB nodes if available.

## Environment variables

| Variable | Purpose |
|---|---|
| `BLUEBEAR_LLM=1` | Select `transformers` backend in `resolve()` |
| `HF_HOME` | Model cache root (RDS project path) |
| `TRANSFORMERS_MODEL` | Hugging Face model id |
| `HF_TOKEN` | Only for gated models |

Set automatically by `run_llm.sh`; override in `config.env`.

## Troubleshooting

**Import error for pydantic/sklearn** — `pip install --user` after loading the Transformers module.

**No tool calls in probe** — try `Qwen/Qwen2.5-14B-Instruct` or check `logs/llm_*.err` for CUDA OOM.

**HF download fails** — confirm network on compute node; or download on login node with `huggingface-cli download` into `HF_HOME`.

**Data missing** — run `bash download_data.sh`; agent tools need `01_Original/data/metrics.csv` etc.

**Do not run inference on the login node** — always use `sbatch`, `submit.sh`, or `srun` with a GPU.

## Current status

| Item | Status |
|---|---|
| Transformers 4.42 on BlueBEAR | Verified by you |
| CUDA / bbgpu access | Verified |
| Model chosen for this task | Qwen2.5-7B-Instruct |
| Slurm script | `run_llm.sh` + `submit.sh` |
| Python task | `task.py` → `sciops_agent.run_agent(backend="transformers")` |
| HF cache location | **You must set** `HF_HOME` in `config.env` |
| Full account name | **You must set** `SLURM_ACCOUNT` in `config.env` |

After filling `config.env`, run `bash submit.sh --smoke-test`, then `bash submit.sh`.

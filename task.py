#!/usr/bin/env python
"""
BlueBEAR entry point for the SCI OPS agentic data-selection task.

Run as a Slurm batch job (not on the login node):

    sbatch run_llm.sh

Or interactively on a GPU node after requesting one:

    srun --gres=gpu:a100:1 --mem=64G --cpus-per-task=8 --pty bash
    module load bear-apps/2023a
    module load Transformers/4.42.0-foss-2023a-CUDA-12.1.1
    export BLUEBEAR_LLM=1
    export HF_HOME=/rds/projects/<PROJECT>/rym386/huggingface
    python task.py

Modes:
    python task.py                  full agent run (default)
    python task.py --smoke-test     verify GPU + model load only
    python task.py --check-data      verify SCI OPS CSV layout
    python task.py --generate       print a short reply from Qwen
    python task.py --probe-tools    one tool-calling turn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"


def _ensure_dirs():
    OUTPUTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)


def smoke_test(model: str | None = None) -> int:
    """Load the model and print environment diagnostics."""
    import torch
    import llm_backends as B

    print("=== BlueBEAR smoke test ===")
    print(f"HF_HOME          : {os.environ.get('HF_HOME', '(default ~/.cache)')}")
    print(f"TRANSFORMERS_MODEL: {model or B.TRANSFORMERS_MODEL}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"VRAM (GB)        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

    B.load_transformers_model(model=model)
    print("\nModel loaded successfully.")
    return 0


def probe_tools(model: str | None = None) -> int:
    """One tool-calling turn to verify the loop can drive inspect_data."""
    import llm_backends as B

    system = "You are a research agent. Call tools to analyse data."
    task = "Summarise the sample before doing anything else."

    def _noop(name: str, args: dict) -> str:
        return json.dumps({"probe": True, "tool": name, "args": args})

    transcript, turns = B.run_transformers_loop(
        task, system, _noop, model=model, max_steps=1, verbose=True)
    print(f"\nProbe finished in {turns} turn(s).")
    if turns == 0:
        print("WARNING: model did not produce a tool call.")
        return 1
    return 0


def generate_reply(prompt: str, *, model: str | None = None, max_new_tokens: int = 128) -> int:
    """Load Qwen and print a short text reply (sanity check)."""
    import torch
    import llm_backends as B

    mdl, tok = B.load_transformers_model(model=model, verbose=True)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    device = next(mdl.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"\nPrompt: {prompt!r}\n")
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("Reply:")
    print(reply.strip())
    return 0


def run_agent_task(task: str, *, model: str | None, max_steps: int,
                   trajectory_id: str | None) -> int:
    """Full SCI OPS agent run; saves trajectory JSON to outputs/."""
    from sciops_agent import DEFAULT_TASK, run_agent
    from trace_schema import Trajectory

    os.environ.setdefault("BLUEBEAR_LLM", "1")
    traj: Trajectory = run_agent(
        task or DEFAULT_TASK,
        backend="transformers",
        model=model,
        max_steps=max_steps,
        trajectory_id=trajectory_id,
        verbose=True,
    )

    out_path = OUTPUTS / f"{traj.trajectory_id}.json"
    out_path.write_text(traj.model_dump_json(indent=2))
    print(f"\nTrajectory saved to {out_path}")

    if traj.outcome:
        summary_path = OUTPUTS / f"{traj.trajectory_id}_summary.txt"
        summary_path.write_text(traj.outcome.final_claim)
        print(f"Summary saved to {summary_path}")

    return 0


def run_experiment(cond: str, *, backend: str = "transformers",
                   model: str | None = None, max_steps: int = 14) -> int:
    """Run one BASELINE/TEST condition through the agent on this machine.

    Identical on a laptop and on a GPU node -- only `backend` changes, so a
    condition debugged locally with backend="scripted" runs unmodified in the
    Slurm job with backend="transformers".
    """
    import run_experiments as E
    import sciops_agent as A

    label, prompt, dataset = E.CONDITIONS[cond]
    ctx = E.context_block(dataset)
    print(f"=== {label} | {dataset} | backend={backend} ===")
    print(ctx)
    print(f"\nPROMPT:\n  {prompt}\n")

    if backend == "scripted":
        from trace_schema import Trajectory, TrajectoryMetadata, now_iso
        A.reset()
        A.TRACE = Trajectory(trajectory_id=f"{cond}_scripted", prompt=prompt,
                             model="scripted",
                             metadata=TrajectoryMetadata(model_version="scripted",
                                                         collection_timestamp=now_iso()))
        traj = A.scripted_correlates(cond)
        traj.recompute_metadata()
    else:
        traj = A.run_agent(task=prompt + "\n\n" + ctx, backend=backend,
                           model=model, max_steps=max_steps,
                           trajectory_id=f"{cond}_{backend}")
    _ensure_dirs()
    out = OUTPUTS / f"{cond}_{backend}_trace.json"
    out.write_text(traj.model_dump_json(indent=1))
    print(f"\n{traj.summary()}")
    print(f"trace -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BlueBEAR LLM task runner")
    parser.add_argument("--smoke-test", action="store_true",
                        help="verify GPU and model loading only")
    parser.add_argument("--check-data", action="store_true",
                        help="verify 01_Original/data CSVs match sciops_agent.py")
    parser.add_argument("--probe-tools", action="store_true",
                        help="run one tool-calling turn")
    parser.add_argument("--generate", action="store_true",
                        help="print a short reply from the model")
    parser.add_argument("--prompt", default="Reply with exactly: Qwen is working on BlueBEAR.",
                        help="prompt for --generate")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--task", default=None,
                        help="research question (default: gad7 vs accuracy)")
    parser.add_argument("--model", default=os.environ.get("TRANSFORMERS_MODEL"),
                        help="Hugging Face model id")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--trajectory-id", default=None)
    parser.add_argument("--experiment", choices=["baseline", "test1", "test2",
                                                 "test3", "test4", "test5"],
                        help="run one BASELINE/TEST condition instead of --task")
    parser.add_argument("--all-experiments", action="store_true",
                        help="run every BASELINE/TEST condition in sequence")
    args = parser.parse_args()

    _ensure_dirs()

    if args.all_experiments or args.experiment:
        import run_experiments as E
        backend = os.environ.get(
            "SCIOPS_BACKEND",
            "transformers" if os.environ.get("BLUEBEAR_LLM") else "scripted")
        conds = list(E.CONDITIONS) if args.all_experiments else [args.experiment]
        rc = 0
        for c in conds:
            rc |= run_experiment(c, backend=backend, model=args.model,
                                 max_steps=args.max_steps)
        return rc
    if args.check_data:
        from check_data import main as check_main
        return check_main()
    if args.smoke_test:
        return smoke_test(args.model)
    if args.generate:
        return generate_reply(args.prompt, model=args.model,
                              max_new_tokens=args.max_new_tokens)
    if args.probe_tools:
        return probe_tools(args.model)
    return run_agent_task(args.task, model=args.model, max_steps=args.max_steps,
                          trajectory_id=args.trajectory_id)


if __name__ == "__main__":
    sys.exit(main())

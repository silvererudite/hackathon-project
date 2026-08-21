#!/usr/bin/env python3
"""Local web interface for running and inspecting SCI OPS agent traces."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

HERE = Path(__file__).resolve().parent
TRACE_DIR = HERE / "traces"
TRACE_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(HERE / "web"), static_url_path="")
RUNS: dict[str, dict] = {}
LOCK = threading.Lock()


def _runtime_sources() -> dict[str, str]:
    return {
        "Restricted execution policy · code_runner.py":
            (HERE / "code_runner.py").read_text(encoding="utf-8"),
        "ReAct generation loop · code_agent.py":
            (HERE / "code_agent.py").read_text(encoding="utf-8"),
    }


def _run_agent(run_id: str, prompt: str, backend: str, max_steps: int) -> None:
    output = TRACE_DIR / f"web-{run_id}.json"
    command = [
        sys.executable,
        "-u",
        str(HERE / "agent_cli.py"),
        prompt,
        "--backend",
        backend,
        "--max-steps",
        str(max_steps),
        "--output",
        str(output),
    ]
    with LOCK:
        RUNS[run_id]["status"] = "running"

    try:
        process = subprocess.Popen(
            command,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            with LOCK:
                RUNS[run_id]["logs"].append(line.rstrip())
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Agent process exited with status {return_code}.")
        trajectory = json.loads(output.read_text(encoding="utf-8"))
        with LOCK:
            RUNS[run_id].update(
                status="completed",
                trajectory=trajectory,
                output=str(output),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
    except Exception as exc:
        with LOCK:
            RUNS[run_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/runtime")
def runtime():
    return jsonify({"sources": _runtime_sources()})


@app.get("/figures/<path:filename>")
def figure(filename: str):
    return send_from_directory(HERE / "figures", filename)


@app.post("/api/runs")
def create_run():
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    backend = str(body.get("backend", "auto"))
    try:
        max_steps = int(body.get("max_steps", 12))
    except (TypeError, ValueError):
        return jsonify({"error": "max_steps must be an integer."}), 400
    if not prompt:
        return jsonify({"error": "Enter a research question before running the agent."}), 400
    if len(prompt) > 10_000:
        return jsonify({"error": "The prompt must be no longer than 10,000 characters."}), 400
    if backend not in {"auto", "openai", "aitta", "anthropic", "huggingface", "scripted"}:
        return jsonify({"error": "Unsupported backend."}), 400
    if not 1 <= max_steps <= 25:
        return jsonify({"error": "max_steps must be between 1 and 25."}), 400

    run_id = uuid.uuid4().hex[:12]
    with LOCK:
        RUNS[run_id] = {
            "id": run_id,
            "status": "queued",
            "logs": [],
            "prompt": prompt,
            "backend": backend,
            "max_steps": max_steps,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    threading.Thread(
        target=_run_agent,
        args=(run_id, prompt, backend, max_steps),
        daemon=True,
    ).start()
    return jsonify({"run_id": run_id}), 202


@app.get("/api/runs/<run_id>")
def get_run(run_id: str):
    try:
        after = max(0, int(request.args.get("after", 0)))
    except ValueError:
        after = 0
    with LOCK:
        run = RUNS.get(run_id)
        if not run:
            abort(404)
        payload = {
            "id": run_id,
            "status": run["status"],
            "logs": run["logs"][after:],
            "next": len(run["logs"]),
            "error": run.get("error"),
        }
        if run["status"] == "completed":
            payload["trajectory"] = run["trajectory"]
            payload["download_url"] = f"/api/runs/{run_id}/download"
    return jsonify(payload)


@app.get("/api/runs/<run_id>/download")
def download_run(run_id: str):
    with LOCK:
        run = RUNS.get(run_id)
        if not run or run.get("status") != "completed":
            abort(404)
        output = Path(run["output"])
    return send_file(output, as_attachment=True, download_name=output.name)


def main() -> None:
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()

"""Compatibility import for the tool-free generated-code ReAct agent.

New code should import :mod:`code_agent` directly. This module remains so the
existing notebook and rating UI continue to work without tool definitions.
"""
from code_agent import (
    DEFAULT_TASK,
    _rated_from_disk,
    build_system_prompt,
    execute_code,
    run_agent,
)

__all__ = [
    "DEFAULT_TASK", "_rated_from_disk", "build_system_prompt",
    "execute_code", "run_agent",
]

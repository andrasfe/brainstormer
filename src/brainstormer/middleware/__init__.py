"""Middleware and hooks system for Brainstormer."""

from .hooks import (
    Hook,
    HookManager,
    HookPhase,
    HookResult,
    hook,
)
from .lifecycle import (
    AgentCompletionMiddleware,
    AgentSpawnMiddleware,
    LifecycleMiddleware,
    PlanCreationMiddleware,
    ResearchWriteMiddleware,
    SearchMiddleware,
)

__all__ = [
    "AgentCompletionMiddleware",
    "AgentSpawnMiddleware",
    "Hook",
    "HookManager",
    "HookPhase",
    "HookResult",
    "LifecycleMiddleware",
    "PlanCreationMiddleware",
    "ResearchWriteMiddleware",
    "SearchMiddleware",
    "hook",
]

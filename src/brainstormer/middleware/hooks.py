"""Hook system for extensible middleware."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from ..utils.logging import get_logger

logger = get_logger(__name__)


class HookPhase(str, Enum):
    """Phases when hooks can be executed."""

    PRE = "pre"
    POST = "post"


@dataclass
class HookResult:
    """Result from a hook execution."""

    success: bool
    modified_data: Any = None
    error: str | None = None
    should_abort: bool = False


@dataclass
class Hook:
    """A registered hook."""

    name: str
    event: str
    phase: HookPhase
    handler: Callable
    priority: int = 0  # Lower runs first
    enabled: bool = True

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise ValueError(f"Hook handler must be callable: {self.name}")


class HookManager:
    """Manages registration and execution of hooks."""

    # Event types supported by the system
    EVENTS: ClassVar[set[str]] = {
        "plan_creation",
        "agent_spawn",
        "research_write",
        "search",
        "completion",
        "session_start",
        "session_end",
        "memory_store",
        "memory_recall",
        "skill_load",
        "tool_call",
    }

    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = {event: [] for event in self.EVENTS}
        self._hook_results: list[dict[str, Any]] = []

    def register(
        self,
        event: str,
        handler: Callable,
        phase: HookPhase = HookPhase.PRE,
        name: str | None = None,
        priority: int = 0,
    ) -> Hook:
        """Register a hook for an event."""
        if event not in self.EVENTS:
            raise ValueError(f"Unknown event: {event}. Valid events: {self.EVENTS}")

        hook = Hook(
            name=name or f"{event}_{phase.value}_{len(self._hooks[event])}",
            event=event,
            phase=phase,
            handler=handler,
            priority=priority,
        )
        self._hooks[event].append(hook)
        self._hooks[event].sort(key=lambda h: h.priority)

        logger.debug(f"Registered hook: {hook.name} for {event} ({phase.value})")
        return hook

    def unregister(self, hook: Hook) -> bool:
        """Unregister a hook."""
        if hook.event in self._hooks and hook in self._hooks[hook.event]:
            self._hooks[hook.event].remove(hook)
            return True
        return False

    async def execute(
        self,
        event: str,
        phase: HookPhase,
        data: Any,
        context: dict | None = None,
    ) -> tuple[Any, list[HookResult]]:
        """Execute all hooks for an event phase."""
        results = []
        current_data = data

        hooks = [h for h in self._hooks.get(event, []) if h.phase == phase and h.enabled]

        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook.handler):
                    result = await hook.handler(current_data, context or {})
                else:
                    result = hook.handler(current_data, context or {})

                if isinstance(result, HookResult):
                    hook_result = result
                else:
                    hook_result = HookResult(success=True, modified_data=result)

                if hook_result.modified_data is not None:
                    current_data = hook_result.modified_data

                results.append(hook_result)

                # Log hook execution
                self._hook_results.append({
                    "hook": hook.name,
                    "event": event,
                    "phase": phase.value,
                    "success": hook_result.success,
                    "error": hook_result.error,
                })

                if hook_result.should_abort:
                    logger.warning(f"Hook {hook.name} requested abort for {event}")
                    break

            except Exception as e:
                logger.error(f"Hook {hook.name} failed: {e}")
                results.append(HookResult(success=False, error=str(e)))

        return current_data, results

    async def execute_pre(
        self, event: str, data: Any, context: dict | None = None
    ) -> tuple[Any, list[HookResult]]:
        """Execute pre-event hooks."""
        return await self.execute(event, HookPhase.PRE, data, context)

    async def execute_post(
        self, event: str, data: Any, context: dict | None = None
    ) -> tuple[Any, list[HookResult]]:
        """Execute post-event hooks."""
        return await self.execute(event, HookPhase.POST, data, context)

    def get_hooks(self, event: str | None = None) -> list[Hook]:
        """Get registered hooks, optionally filtered by event."""
        if event:
            return self._hooks.get(event, [])
        return [hook for hooks in self._hooks.values() for hook in hooks]

    def get_results(self) -> list[dict]:
        """Get execution results log."""
        return self._hook_results.copy()

    def clear_results(self) -> None:
        """Clear the results log."""
        self._hook_results.clear()


def hook(
    event: str,
    phase: HookPhase = HookPhase.PRE,
    name: str | None = None,
    priority: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to mark a function as a hook."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._hook_config = {  # type: ignore[attr-defined]
            "event": event,
            "phase": phase,
            "name": name or func.__name__,
            "priority": priority,
        }
        return func

    return decorator


def load_hooks_from_file(file_path: Path, manager: HookManager) -> list[Hook]:
    """Load hooks defined in a Python file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("hooks_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load hooks from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registered = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "_hook_config"):
            config = obj._hook_config
            hook = manager.register(
                event=config["event"],
                handler=obj,
                phase=config["phase"],
                name=config["name"],
                priority=config["priority"],
            )
            registered.append(hook)
            logger.info(f"Loaded hook from file: {hook.name}")

    return registered

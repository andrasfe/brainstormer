"""Lifecycle middleware integrating with DeepAgents."""

from dataclasses import dataclass, field
from typing import Any

from ..backends.memory import MemoryManager
from ..backends.persistence import PersistenceManager
from ..utils.logging import get_logger
from .hooks import HookManager

logger = get_logger(__name__)


@dataclass
class MiddlewareContext:
    """Context passed through middleware chain."""

    session_id: str
    hook_manager: HookManager
    persistence: PersistenceManager
    memory: MemoryManager | None = None
    metadata: dict[str, Any] | None = field(default=None)


class LifecycleMiddleware:
    """Base class for lifecycle middleware."""

    def __init__(self, context: MiddlewareContext):
        self.context = context
        self.hook_manager = context.hook_manager

    async def before(self, data: Any) -> Any:
        """Called before the operation."""
        return data

    async def after(self, data: Any, result: Any) -> Any:
        """Called after the operation."""
        return result


class PlanCreationMiddleware(LifecycleMiddleware):
    """Middleware for plan creation events."""

    async def before(self, plan_data: dict[str, Any]) -> dict[str, Any]:
        """Pre-process plan creation."""
        data, results = await self.hook_manager.execute_pre(
            "plan_creation",
            plan_data,
            {"session_id": self.context.session_id},
        )

        # Check for abort
        for result in results:
            if result.should_abort:
                raise RuntimeError("Plan creation aborted by hook")

        return dict(data) if data else plan_data

    async def after(self, plan_data: dict, plan_content: str) -> str:
        """Post-process plan creation."""
        # Store plan in persistence
        plan_path = self.context.persistence.write_plan(
            self.context.session_id, plan_content
        )

        # Execute post hooks
        result, _ = await self.hook_manager.execute_post(
            "plan_creation",
            {"plan": plan_content, "path": str(plan_path)},
            {"session_id": self.context.session_id},
        )

        # Store in long-term memory if available
        if self.context.memory:
            self.context.memory.remember_insight(
                content=f"Research Plan:\n{plan_content}",
                session_id=self.context.session_id,
                source="plan_creation",
            )

        logger.info(f"Plan created and saved to {plan_path}")
        return result.get("plan", plan_content) if isinstance(result, dict) else plan_content


class AgentSpawnMiddleware(LifecycleMiddleware):
    """Middleware for agent spawn events."""

    async def before(self, agent_config: dict[str, Any]) -> dict[str, Any]:
        """Pre-process agent spawn."""
        data, results = await self.hook_manager.execute_pre(
            "agent_spawn",
            agent_config,
            {"session_id": self.context.session_id},
        )

        for result in results:
            if result.should_abort:
                raise RuntimeError(f"Agent spawn aborted by hook: {agent_config.get('name')}")

        return dict(data) if data else agent_config

    async def after(self, agent_config: dict, agent_id: str) -> str:
        """Post-process agent spawn."""
        # Record agent in persistence
        self.context.persistence.store.create_agent_state(
            agent_id=agent_id,
            session_id=self.context.session_id,
            agent_name=agent_config.get("name", "unnamed"),
            focus_area=agent_config.get("focus_area", ""),
            state_data=agent_config,
        )

        await self.hook_manager.execute_post(
            "agent_spawn",
            {"agent_id": agent_id, "config": agent_config},
            {"session_id": self.context.session_id},
        )

        logger.info(f"Agent spawned: {agent_config.get('name')} ({agent_id})")
        return agent_id


class ResearchWriteMiddleware(LifecycleMiddleware):
    """Middleware for research output write events."""

    async def before(self, write_data: dict[str, Any]) -> dict[str, Any]:
        """Pre-process research write."""
        data, _ = await self.hook_manager.execute_pre(
            "research_write",
            write_data,
            {"session_id": self.context.session_id},
        )
        return dict(data) if data else write_data

    async def after(self, write_data: dict, file_path: str) -> str:
        """Post-process research write."""
        # Store in long-term memory
        if self.context.memory and write_data.get("content"):
            self.context.memory.remember_research(
                session_id=self.context.session_id,
                agent_name=write_data.get("agent_name", "unknown"),
                content=write_data["content"],
                focus_area=write_data.get("focus_area", ""),
                tags=write_data.get("tags", []),
            )

        await self.hook_manager.execute_post(
            "research_write",
            {"path": file_path, "data": write_data},
            {"session_id": self.context.session_id},
        )

        logger.debug(f"Research written to {file_path}")
        return file_path


class SearchMiddleware(LifecycleMiddleware):
    """Middleware for web search events."""

    async def before(self, search_query: dict[str, Any]) -> dict[str, Any]:
        """Pre-process search query."""
        data, _ = await self.hook_manager.execute_pre(
            "search",
            search_query,
            {"session_id": self.context.session_id},
        )
        return dict(data) if data else search_query

    async def after(self, search_query: dict, results: list) -> list:
        """Post-process search results."""
        result, _ = await self.hook_manager.execute_post(
            "search",
            {"query": search_query, "results": results},
            {"session_id": self.context.session_id},
        )

        # Cache search results in memory
        if self.context.memory and results:
            summary = f"Search: {search_query.get('query', '')}\nResults: {len(results)} found"
            self.context.memory.remember_insight(
                content=summary,
                session_id=self.context.session_id,
                source="web_search",
            )

        return result.get("results", results) if isinstance(result, dict) else results


class AgentCompletionMiddleware(LifecycleMiddleware):
    """Middleware for agent completion events."""

    async def before(self, completion_data: dict[str, Any]) -> dict[str, Any]:
        """Pre-process completion."""
        data, _ = await self.hook_manager.execute_pre(
            "completion",
            completion_data,
            {"session_id": self.context.session_id},
        )
        return dict(data) if data else completion_data

    async def after(self, completion_data: dict, result: Any) -> Any:
        """Post-process completion."""
        # Update agent state
        agent_id = completion_data.get("agent_id")
        if agent_id:
            self.context.persistence.store.update_agent_state(
                agent_id,
                status="completed",
                result_path=completion_data.get("result_path"),
            )

        await self.hook_manager.execute_post(
            "completion",
            {"completion_data": completion_data, "result": result},
            {"session_id": self.context.session_id},
        )

        logger.info(f"Agent completed: {completion_data.get('agent_name', 'unknown')}")
        return result

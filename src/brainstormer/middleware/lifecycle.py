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


@dataclass
class QualityMetrics:
    """Tracks quality metrics for a research session."""

    search_count: int = 0
    citation_count: int = 0
    phases_completed: list[str] = field(default_factory=list)
    confidence_ratings_found: int = 0
    word_count: int = 0
    sources_cited: set[str] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "search_count": self.search_count,
            "citation_count": self.citation_count,
            "phases_completed": self.phases_completed,
            "confidence_ratings_found": self.confidence_ratings_found,
            "word_count": self.word_count,
            "sources_cited": list(self.sources_cited),
            "issues": self.issues,
        }


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for research."""

    min_searches_phase1: int = 5
    min_searches_phase2: int = 10
    min_searches_total: int = 15
    min_citations_per_phase: int = 3
    min_word_count_per_phase: int = 500
    require_confidence_ratings: bool = True
    require_counterarguments: bool = True


class QualityGateMiddleware(LifecycleMiddleware):
    """Middleware that enforces research quality standards.

    Tracks metrics across the research session and validates
    that quality thresholds are met before allowing phase transitions.
    """

    # Class-level storage for session metrics
    _session_metrics: dict[str, QualityMetrics] = {}

    def __init__(
        self,
        context: MiddlewareContext,
        thresholds: QualityThresholds | None = None,
    ):
        super().__init__(context)
        self.thresholds = thresholds or QualityThresholds()

        # Initialize metrics for this session
        if context.session_id not in self._session_metrics:
            self._session_metrics[context.session_id] = QualityMetrics()

    @property
    def metrics(self) -> QualityMetrics:
        """Get metrics for current session."""
        return self._session_metrics[self.context.session_id]

    def record_search(self, query: str) -> None:
        """Record a search was performed."""
        self.metrics.search_count += 1
        logger.debug(f"Quality: Search #{self.metrics.search_count}: {query[:50]}...")

    def record_write(self, content: str, file_path: str) -> None:
        """Analyze written content for quality metrics."""
        import re

        # Count words
        self.metrics.word_count += len(content.split())

        # Detect phase completion
        phase_patterns = [
            ("phase1", r"phase1|phase_1|initial.?research"),
            ("phase2", r"phase2|phase_2|deep.?dive"),
            ("phase3", r"phase3|phase_3|critical.?review"),
            ("phase4", r"phase4|phase_4|synthesis"),
            ("final", r"final.?report|FINAL_REPORT"),
        ]
        for phase_name, pattern in phase_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                if phase_name not in self.metrics.phases_completed:
                    self.metrics.phases_completed.append(phase_name)
                    logger.info(f"Quality: Phase '{phase_name}' completed")

        # Count citations (URLs, references)
        url_pattern = r'https?://[^\s\)\]\"\'<>]+'
        urls = re.findall(url_pattern, content)
        self.metrics.citation_count += len(urls)
        self.metrics.sources_cited.update(urls)

        # Check for confidence ratings
        confidence_pattern = r'\b(high|medium|low)\s*confidence\b|\bconfidence[:\s]*(high|medium|low)\b'
        confidence_matches = re.findall(confidence_pattern, content, re.IGNORECASE)
        self.metrics.confidence_ratings_found += len(confidence_matches)

    def validate_phase_transition(self, from_phase: str, to_phase: str) -> tuple[bool, list[str]]:
        """Validate if transition between phases meets quality standards."""
        issues = []

        if from_phase == "phase1":
            if self.metrics.search_count < self.thresholds.min_searches_phase1:
                issues.append(
                    f"Phase 1 requires at least {self.thresholds.min_searches_phase1} searches, "
                    f"found {self.metrics.search_count}"
                )

        if to_phase == "final":
            if self.metrics.search_count < self.thresholds.min_searches_total:
                issues.append(
                    f"Research requires at least {self.thresholds.min_searches_total} total searches, "
                    f"found {self.metrics.search_count}"
                )
            if self.thresholds.require_confidence_ratings and self.metrics.confidence_ratings_found == 0:
                issues.append("Final report should include confidence ratings for conclusions")

        self.metrics.issues.extend(issues)
        return len(issues) == 0, issues

    def get_quality_report(self) -> dict[str, Any]:
        """Generate a quality assessment report."""
        metrics = self.metrics

        # Calculate quality score (0-100)
        score = 0
        max_score = 100

        # Search coverage (30 points)
        search_ratio = min(metrics.search_count / self.thresholds.min_searches_total, 1.0)
        score += int(search_ratio * 30)

        # Citation density (25 points)
        if metrics.word_count > 0:
            citations_per_1000_words = (metrics.citation_count / metrics.word_count) * 1000
            citation_score = min(citations_per_1000_words / 5, 1.0)  # Target: 5 citations per 1000 words
            score += int(citation_score * 25)

        # Phase completion (25 points)
        expected_phases = ["phase1", "phase2", "phase3", "phase4", "final"]
        phase_ratio = len([p for p in expected_phases if p in metrics.phases_completed]) / len(expected_phases)
        score += int(phase_ratio * 25)

        # Confidence ratings (10 points)
        if metrics.confidence_ratings_found >= 3:
            score += 10
        elif metrics.confidence_ratings_found > 0:
            score += 5

        # Unique sources (10 points)
        if len(metrics.sources_cited) >= 10:
            score += 10
        elif len(metrics.sources_cited) >= 5:
            score += 5

        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": score,
            "max_score": max_score,
            "grade": grade,
            "metrics": metrics.to_dict(),
            "thresholds": {
                "min_searches_total": self.thresholds.min_searches_total,
                "min_citations_per_phase": self.thresholds.min_citations_per_phase,
            },
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> list[str]:
        """Generate recommendations for improving research quality."""
        recommendations = []
        metrics = self.metrics

        if metrics.search_count < self.thresholds.min_searches_total:
            recommendations.append(
                f"Conduct more web searches ({self.thresholds.min_searches_total - metrics.search_count} more needed)"
            )

        if metrics.citation_count < 10:
            recommendations.append("Include more source citations to support claims")

        if metrics.confidence_ratings_found == 0:
            recommendations.append("Add confidence ratings (High/Medium/Low) to conclusions")

        if len(metrics.sources_cited) < 5:
            recommendations.append("Diversify sources - cite from multiple domains")

        if "phase3" not in metrics.phases_completed:
            recommendations.append("Complete critical review phase to challenge assumptions")

        return recommendations

    async def before(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pre-process - check if quality gates allow proceeding."""
        event_type = data.get("event_type", "")

        # Track searches
        if event_type == "search":
            self.record_search(data.get("query", ""))

        return data

    async def after(self, data: dict[str, Any], result: Any) -> Any:
        """Post-process - record metrics and validate quality."""
        event_type = data.get("event_type", "")

        # Track file writes
        if event_type == "write":
            self.record_write(
                content=data.get("content", ""),
                file_path=data.get("file_path", ""),
            )

        # On session end, generate quality report
        if event_type == "session_end":
            report = self.get_quality_report()
            logger.info(f"Quality Report - Score: {report['score']}/100 (Grade: {report['grade']})")

            # Store report in persistence
            if self.context.persistence:
                session_dir = self.context.persistence.get_session_dir(self.context.session_id)
                report_path = session_dir / "QUALITY_REPORT.json"
                import json
                report_path.write_text(json.dumps(report, indent=2))
                logger.info(f"Quality report saved to {report_path}")

        return result

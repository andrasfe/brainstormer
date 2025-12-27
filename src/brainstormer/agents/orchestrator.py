"""Main orchestrator agent for research coordination."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ..backends.memory import ChromaMemoryStore, MemoryManager
from ..backends.persistence import PersistenceManager
from ..config import Settings
from ..middleware.hooks import HookManager
from ..middleware.lifecycle import (
    AgentCompletionMiddleware,
    AgentSpawnMiddleware,
    MiddlewareContext,
    PlanCreationMiddleware,
    ResearchWriteMiddleware,
    SearchMiddleware,
)
from ..skills.loader import SkillRegistry
from ..utils.logging import get_logger
from .subagents import SubagentConfig, SubagentManager
from .tools import create_file_context_tool, create_memory_tools, create_search_tool

# Type alias for the backend factory
BackendFactory = CompositeBackend

logger = get_logger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """You are a research orchestrator agent. Your role is to:

1. **Understand the Research Problem**: Analyze the problem statement and any provided input files/context
2. **Create a Research Plan**: Break down the problem into distinct focus areas for investigation
3. **Delegate to Subagents**: Assign each focus area to a specialized research subagent
4. **Synthesize Results**: Collect and synthesize findings from all subagents
5. **Write the Research Plan**: Document the plan and agent assignments in RESEARCH_PLAN.md

## Research Plan Format

Write your research plan as a markdown file with:

```markdown
# Research Plan: [Problem Title]

## Problem Statement
[Clear description of what we're researching]

## Input Context
[Summary of provided files/pointers]

## Focus Areas

### 1. [Focus Area Name]
- **Agent**: [agent-name]
- **Objective**: [What this agent will investigate]
- **Key Questions**: [Specific questions to answer]

### 2. [Focus Area Name]
...

## Timeline & Dependencies
[Any sequencing or dependencies between research areas]

## Expected Outputs
[What the final deliverables should include]
```

## Guidelines

- Create 2-5 focus areas depending on problem complexity
- Each focus area should be distinct but collectively cover the problem
- Use the `task` tool to spawn subagents for each focus area
- Subagents will write their findings to their own subdirectories
- After all subagents complete, synthesize findings into a final report

## Available Input Context

Use the `get_input_context` tool to access files provided by the user.
Use `internet_search` for web research when needed.
"""


class ResearchOrchestrator:
    """Orchestrates research sessions using DeepAgents."""

    def __init__(
        self,
        settings: Settings,
        output_dir: Path,
        skills_registry: SkillRegistry | None = None,
        subagent_manager: SubagentManager | None = None,
        hook_manager: HookManager | None = None,
    ):
        self.settings = settings
        self.output_dir = output_dir
        self.skills_registry = skills_registry
        self.subagent_manager = subagent_manager
        self.hook_manager = hook_manager or HookManager()

        # Initialize persistence
        self.persistence = PersistenceManager(
            db_path=settings.sqlite_db_path,
            base_output_dir=output_dir,
        )

        # Initialize memory
        self.chroma_store = ChromaMemoryStore(
            persist_directory=settings.chromadb_path,
        )
        self.memory_manager = MemoryManager(self.chroma_store)

        # Initialize model
        self.model = init_chat_model(settings.get_model_string())

        logger.info(f"Initialized ResearchOrchestrator with model: {settings.get_model_string()}")

    def _create_subagents_config(self, focus_areas: list[str]) -> list[dict]:
        """Create subagent configurations for focus areas."""
        subagents = []

        for focus_area in focus_areas:
            # Try to find a matching configured subagent
            if self.subagent_manager:
                matches = self.subagent_manager.match_for_focus(focus_area)
                if matches:
                    config = matches[0]
                    subagents.append(config.to_deepagent_config())
                    continue

            # Create dynamic subagent
            dynamic_config = SubagentConfig(
                name=f"research-{focus_area.lower().replace(' ', '-')[:20]}",
                description=f"Research agent for: {focus_area}",
                system_prompt=f"""You are a research subagent focused on: {focus_area}

Your task is to:
1. Research this specific area thoroughly
2. Use web search to find relevant information
3. Analyze and synthesize your findings
4. Write a comprehensive report to your output directory

Write your findings to markdown files in your working directory.
Create subdirectories if the research becomes complex with multiple subtopics.
""",
            )
            subagents.append(dynamic_config.to_deepagent_config())

        return subagents

    async def run_research(
        self,
        problem: str,
        input_files: list[dict] | None = None,
        focus_areas: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        """
        Run a research session.

        Args:
            problem: The research problem/question
            input_files: Parsed input files (from file_parser)
            focus_areas: Optional predefined focus areas
            session_id: Optional session ID (generated if not provided)

        Returns:
            Research session results
        """
        session_id = session_id or f"research-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        # Create session in persistence
        self.persistence.store.create_session(
            session_id=session_id,
            problem=problem,
            metadata={
                "input_files": [f["name"] for f in (input_files or [])],
                "start_time": datetime.now(tz=UTC).isoformat(),
            },
        )

        # Create middleware context
        middleware_context = MiddlewareContext(
            session_id=session_id,
            hook_manager=self.hook_manager,
            persistence=self.persistence,
            memory=self.memory_manager,
        )

        # Initialize middleware (stored for potential future use in hook integration)
        _ = PlanCreationMiddleware(middleware_context)
        _ = AgentSpawnMiddleware(middleware_context)
        _ = ResearchWriteMiddleware(middleware_context)
        _ = SearchMiddleware(middleware_context)
        _ = AgentCompletionMiddleware(middleware_context)

        # Create tools (using Any for heterogeneous callable types)
        tools: list[Any] = []

        # Add search tool
        if self.settings.tavily_api_key:
            tools.append(create_search_tool(self.settings.tavily_api_key))

        # Add memory tools
        memory_tools = create_memory_tools(self.memory_manager)
        tools.extend(memory_tools.values())

        # Add input context tool
        if input_files:
            tools.append(create_file_context_tool(input_files))

        # Build system prompt
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT

        # Add skills to prompt
        if self.skills_registry:
            skills_prompt = self.skills_registry.get_combined_prompt()
            if skills_prompt:
                system_prompt += f"\n\n## Available Skills\n\n{skills_prompt}"

        # Add input file context to prompt
        if input_files:
            files_context = "\n".join([
                f"- {f['name']} ({f['type']}, {f['size']} bytes)"
                for f in input_files
            ])
            system_prompt += f"\n\n## Input Files\n\nThe following files have been provided:\n{files_context}\n\nUse the `get_input_context` tool to read their contents."

        # Create subagent configurations
        subagents = []
        if focus_areas:
            subagents = self._create_subagents_config(focus_areas)
        elif self.subagent_manager:
            # Use all configured subagents
            subagents = [c.to_deepagent_config() for c in self.subagent_manager.list_all()]

        # Create the deep agent
        checkpointer = MemorySaver()
        store = InMemoryStore()

        def make_backend(runtime: object) -> CompositeBackend:
            return CompositeBackend(
                default=StateBackend(runtime),
                routes={
                    "/memories/": StoreBackend(runtime),
                },
            )

        agent = create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt,
            subagents=subagents if subagents else None,
            store=store,
            backend=make_backend,
            checkpointer=checkpointer,
        )

        # Execute pre-session hooks
        await self.hook_manager.execute_pre(
            "session_start",
            {"session_id": session_id, "problem": problem},
        )

        # Run the agent
        logger.info(f"Starting research session: {session_id}")

        config = {"configurable": {"thread_id": session_id}}
        initial_message = f"""# Research Request

## Problem Statement
{problem}

Please:
1. Analyze this research problem
2. Create a comprehensive research plan with focus areas
3. Write the plan to RESEARCH_PLAN.md
4. Spawn subagents for each focus area
5. Synthesize the findings when all subagents complete
"""

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": initial_message}]},
            config=config,
        )

        # Execute post-session hooks
        await self.hook_manager.execute_post(
            "session_end",
            {"session_id": session_id, "result": result},
        )

        # Update session status
        self.persistence.store.update_session(
            session_id,
            status="completed",
            metadata={
                "end_time": datetime.now(tz=UTC).isoformat(),
                "message_count": len(result.get("messages", [])),
            },
        )

        logger.info(f"Research session completed: {session_id}")

        return {
            "session_id": session_id,
            "output_dir": str(self.persistence.get_session_dir(session_id)),
            "result": result,
        }

    async def resume_session(self, session_id: str, message: str) -> dict:
        """Resume an existing research session with a new message."""
        session = self.persistence.store.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Recreate the agent with same configuration
        # This would need to reload the checkpointed state
        # For now, we'll create a continuation message

        logger.info(f"Resuming session: {session_id}")

        # This is a simplified version - full implementation would
        # reload the agent state from the checkpointer
        return {
            "session_id": session_id,
            "message": "Session resumption requires full state reload",
        }

    def get_session_status(self, session_id: str) -> dict | None:
        """Get the status of a research session."""
        session = self.persistence.store.get_session(session_id)
        if not session:
            return None

        agents = self.persistence.store.get_session_agents(session_id)
        artifacts = self.persistence.store.get_session_artifacts(session_id)

        return {
            "session": session,
            "agents": agents,
            "artifacts": artifacts,
            "output_dir": str(self.persistence.get_session_dir(session_id)),
        }

    def list_sessions(self, status: str | None = None) -> list[dict]:
        """List all research sessions."""
        return self.persistence.store.list_sessions(status)

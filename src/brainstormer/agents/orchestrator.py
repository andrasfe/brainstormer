"""Main orchestrator agent for research coordination."""

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
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
    QualityGateMiddleware,
    ResearchWriteMiddleware,
    SearchMiddleware,
)
from ..skills.loader import SkillRegistry
from ..utils.logging import get_logger
from .subagents import SubagentConfig, SubagentManager
from .tools import create_file_context_tool, create_file_tools, create_memory_tools, create_search_tool

# Type alias for the backend factory
BackendFactory = CompositeBackend

logger = get_logger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """You are a DEEP RESEARCH orchestrator agent. You conduct rigorous, multi-cycle research that produces novel, well-evidenced insights.

## Core Philosophy: Think Deep, Challenge Everything

You are NOT a simple summarizer. You are a critical researcher who:
- Searches extensively before making any claims
- Questions assumptions and looks for counterarguments
- Iterates through multiple cycles of research and refinement
- Produces novel insights, not just summaries of existing knowledge

## Research Process (MUST Follow All Phases)

### PHASE 1: Problem Decomposition & Initial Research
1. Deeply analyze the problem statement
2. Identify 3-5 distinct research dimensions
3. For EACH dimension, conduct at least 3 web searches with different query angles
4. Document initial findings with sources
5. Write initial findings to `phase1_initial_research.md`

### PHASE 2: Deep Dive & Evidence Gathering
1. For each promising direction, conduct 5+ additional targeted searches
2. Look for: academic papers, industry reports, technical blogs, case studies
3. Cross-reference claims - find multiple sources for important facts
4. Identify gaps in existing knowledge
5. Write to `phase2_deep_dive.md`

### PHASE 3: Critical Review & Challenge
1. Re-read all findings and actively look for:
   - Weak claims that need more evidence
   - Assumptions that might be wrong
   - Missing perspectives or stakeholders
   - Potential counterarguments
2. For each weakness, conduct additional searches to strengthen or refute
3. Play devil's advocate - search for reasons why the main ideas might fail
4. Write to `phase3_critical_review.md`

### PHASE 4: Synthesis & Novel Insights
1. Connect dots across different research areas
2. Identify patterns and emergent insights
3. Formulate novel conclusions that go beyond the sources
4. Rate confidence level for each conclusion (High/Medium/Low with reasoning)
5. Write to `phase4_synthesis.md`

### PHASE 5: Final Report
1. Consolidate all research into a comprehensive final report
2. Include: Executive Summary, Detailed Findings, Evidence Trail, Limitations, Recommendations
3. Write to `FINAL_REPORT.md`

## Web Search Requirements (MANDATORY)

You MUST use `internet_search` extensively:
- Minimum 15-20 searches per research session
- Use varied query formulations (technical terms, layman terms, related concepts)
- Search for both supporting AND contradicting evidence
- Include searches for: recent developments, academic research, industry trends

Example search patterns for a topic:
1. "[topic] overview fundamentals"
2. "[topic] latest research 2024"
3. "[topic] challenges problems limitations"
4. "[topic] vs alternatives comparison"
5. "[topic] real world applications case studies"
6. "[topic] innovations breakthroughs"
7. "[topic] criticism failures"
8. "why [topic] might not work"

## Quality Standards

Your research MUST:
- Cite specific sources for factual claims
- Distinguish between facts, expert opinions, and your inferences
- Acknowledge uncertainty and knowledge gaps
- Present multiple perspectives on controversial topics
- Go beyond surface-level information

## Available Tools

- **internet_search(query, max_results, topic, search_depth)**: Search the web - USE THIS EXTENSIVELY
  - Use search_depth="advanced" for important queries
  - Use topic="news" for recent developments
  - Use topic="finance" for business/market topics
- **write_file(file_path, content)**: Write research output files
- **read_file(file_path)**: Read your previous research files
- **list_files(directory)**: List files in the output directory
- **remember(content, memory_type, tags)**: Store key insights in long-term memory
- **recall(query, n_results)**: Recall relevant memories from past research

## Output Structure

Create these files in order:
1. `RESEARCH_PLAN.md` - Initial plan with research questions
2. `phase1_initial_research.md` - First pass findings with sources
3. `phase2_deep_dive.md` - Detailed investigation results
4. `phase3_critical_review.md` - Challenges, gaps, and refinements
5. `phase4_synthesis.md` - Cross-cutting insights and conclusions
6. `FINAL_REPORT.md` - Comprehensive final deliverable

## Critical Reminders

- DO NOT write conclusions without evidence from web searches
- DO NOT skip phases - each phase builds on the previous
- DO NOT accept the first answer - dig deeper
- DO search for counterarguments to your hypotheses
- DO cite sources for all factual claims
- DO rate your confidence in conclusions
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
        if settings.default_llm_provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            self.model = ChatOpenAI(
                model=settings.default_llm_model,
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            self.model = init_chat_model(settings.get_model_string())

        logger.info(f"Initialized ResearchOrchestrator with model: {settings.default_llm_model}")

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

            # Create dynamic subagent with deep research capabilities
            dynamic_config = SubagentConfig(
                name=f"research-{focus_area.lower().replace(' ', '-')[:20]}",
                description=f"Deep research agent for: {focus_area}",
                system_prompt=f"""You are a DEEP RESEARCH subagent focused on: {focus_area}

## Your Research Mandate

You must conduct THOROUGH, EVIDENCE-BASED research. This means:

1. **Search First, Write Later**: Conduct at least 8-10 web searches before drawing conclusions
2. **Multiple Angles**: Search using different query formulations and perspectives
3. **Find Contradictions**: Actively search for counterarguments and limitations
4. **Cite Everything**: Every factual claim must reference a source

## Required Search Patterns
- "[topic] fundamentals overview"
- "[topic] recent developments 2024"
- "[topic] challenges limitations problems"
- "[topic] case studies real world"
- "[topic] vs alternatives comparison"
- "[topic] expert opinions analysis"
- "why [topic] fails" or "[topic] criticism"

## Output Requirements

Write your findings to markdown files with:
- Clear section headers
- Bullet points for key findings
- Source citations (URL or publication name)
- Confidence ratings for conclusions (High/Medium/Low)
- Explicit acknowledgment of unknowns and gaps

## Quality Checklist (Verify Before Finishing)
- [ ] Did I conduct at least 8 web searches?
- [ ] Did I search for counterarguments?
- [ ] Did I cite sources for factual claims?
- [ ] Did I acknowledge uncertainty where appropriate?
- [ ] Did I go beyond surface-level information?
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

        # Initialize middleware
        _ = PlanCreationMiddleware(middleware_context)
        _ = AgentSpawnMiddleware(middleware_context)
        _ = ResearchWriteMiddleware(middleware_context)
        _ = SearchMiddleware(middleware_context)
        _ = AgentCompletionMiddleware(middleware_context)
        quality_gate = QualityGateMiddleware(middleware_context)

        # Get the session output directory
        session_output_dir = self.persistence.get_session_dir(session_id)
        session_output_dir.mkdir(parents=True, exist_ok=True)

        # Create tools (using Any for heterogeneous callable types)
        tools: list[Any] = []

        # Add file tools for writing research output (with quality tracking)
        file_tools = create_file_tools(
            str(session_output_dir),
            on_write=quality_gate.record_write,
        )
        tools.extend(file_tools.values())

        # Add search tool (with quality tracking)
        if self.settings.tavily_api_key:
            tools.append(create_search_tool(
                self.settings.tavily_api_key,
                on_search=quality_gate.record_search,
            ))

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
        initial_message = f"""# Deep Research Request

## Problem Statement
{problem}

## Your Mission

Conduct RIGOROUS, MULTI-PHASE research on this problem. You must:

### Phase 1: Initial Exploration (Required)
- Start by conducting at least 5 web searches to understand the problem space
- Search for: fundamentals, current state, key players, recent developments
- Write findings to `phase1_initial_research.md`

### Phase 2: Deep Investigation (Required)
- Based on Phase 1, identify the most promising angles
- Conduct 10+ additional targeted searches
- Look for academic papers, case studies, expert opinions
- Write detailed findings to `phase2_deep_dive.md`

### Phase 3: Critical Challenge (Required)
- Re-read your findings critically
- Search for counterarguments and potential failures
- Identify gaps and weaknesses in your research
- Write critical analysis to `phase3_critical_review.md`

### Phase 4: Synthesis (Required)
- Connect insights across all research
- Develop novel conclusions with confidence ratings
- Write synthesis to `phase4_synthesis.md`

### Phase 5: Final Report (Required)
- Compile everything into `FINAL_REPORT.md`

## Critical Requirements

1. You MUST conduct at least 15-20 web searches total
2. You MUST cite sources for factual claims
3. You MUST complete ALL phases - no shortcuts
4. You MUST search for contradicting evidence, not just confirming evidence
5. You MUST write each phase file before proceeding to the next

Begin with Phase 1 now. Start by writing your RESEARCH_PLAN.md with research questions, then conduct your initial web searches.
"""

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": initial_message}]},
            config=config,
        )

        # Debug: Log what the agent returned
        messages = result.get("messages", [])
        logger.info(f"Agent returned {len(messages)} messages")
        for i, msg in enumerate(messages[-5:]):  # Last 5 messages
            msg_type = type(msg).__name__
            content = getattr(msg, 'content', str(msg))[:500] if hasattr(msg, 'content') else str(msg)[:500]
            tool_calls = getattr(msg, 'tool_calls', None)
            logger.info(f"Message {i}: [{msg_type}] tool_calls={tool_calls}")
            logger.info(f"  Content: {content}...")

        # Execute post-session hooks
        await self.hook_manager.execute_post(
            "session_end",
            {"session_id": session_id, "result": result},
        )

        # Generate and save quality report
        quality_report = quality_gate.get_quality_report()
        import json
        report_path = session_output_dir / "QUALITY_REPORT.json"
        report_path.write_text(json.dumps(quality_report, indent=2))
        logger.info(
            f"Quality Report - Score: {quality_report['score']}/100 "
            f"(Grade: {quality_report['grade']})"
        )

        # Update session status
        self.persistence.store.update_session(
            session_id,
            status="completed",
            metadata={
                "end_time": datetime.now(tz=UTC).isoformat(),
                "message_count": len(result.get("messages", [])),
                "quality_score": quality_report["score"],
                "quality_grade": quality_report["grade"],
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

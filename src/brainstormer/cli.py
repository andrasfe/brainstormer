"""Command-line interface for Brainstormer."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agents.orchestrator import ResearchOrchestrator
from .agents.subagents import SubagentManager, create_default_subagents_file
from .backends.memory import ChromaMemoryStore, MemoryManager
from .config import Settings, load_settings
from .middleware.hooks import HookManager, load_hooks_from_file
from .skills.loader import SkillRegistry, create_skill_directory
from .utils.file_parser import parse_files
from .utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="brainstormer",
    help="Orchestrated research CLI with configurable subagents and long-term memory",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


def get_settings(env_file: Path | None = None) -> Settings:
    """Load and validate settings."""
    settings = load_settings(env_file)
    errors = settings.validate_api_keys()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    return settings


@app.command()
def research(
    problem: Annotated[str, typer.Argument(help="The research problem or question to investigate")],
    files: Annotated[
        list[Path] | None,
        typer.Option("--file", "-f", help="Input files (text or PDF) to include as context"),
    ] = None,
    focus_areas: Annotated[
        list[str] | None,
        typer.Option("--focus", help="Predefined focus areas for the research"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for research results"),
    ] = Path("./research"),
    subagents_file: Annotated[
        Path | None,
        typer.Option("--subagents", "-s", help="JSONL file with subagent configurations"),
    ] = None,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills", help="Directory containing skill definitions"),
    ] = None,
    hooks_file: Annotated[
        Path | None,
        typer.Option("--hooks", help="Python file containing hook definitions"),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option("--env", help="Path to .env file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """
    Start a new research session.

    The orchestrator will analyze your problem, create a research plan,
    delegate to specialized subagents, and synthesize the findings.
    """
    # Setup logging
    setup_logging("DEBUG" if verbose else "INFO")

    # Load settings
    settings = get_settings(env_file)

    # Override paths from settings if provided
    if skills_dir:
        settings.skills_dir = skills_dir

    console.print(Panel.fit(
        f"[bold blue]Brainstormer[/bold blue]\n\n"
        f"[yellow]Problem:[/yellow] {problem[:100]}{'...' if len(problem) > 100 else ''}",
        title="Research Session",
    ))

    # Parse input files
    input_files = []
    if files:
        console.print("\n[bold]Parsing input files...[/bold]")
        input_files = parse_files([Path(f) for f in files])
        for f in input_files:
            console.print(f"  - {f['name']} ({f['type']}, {f['size']} bytes)")

    # Load subagents
    subagent_manager = None
    if subagents_file and subagents_file.exists():
        subagent_manager = SubagentManager(subagents_file)
        console.print(f"\n[bold]Loaded {len(subagent_manager.list_all())} subagent configurations[/bold]")

    # Load skills
    skills_registry = None
    if settings.skills_dir.exists():
        skills_registry = SkillRegistry(settings.skills_dir)
        if skills_registry.list_all():
            console.print(f"\n[bold]Loaded {len(skills_registry.list_all())} skills[/bold]")

    # Load hooks
    hook_manager = HookManager()
    if hooks_file and hooks_file.exists():
        hooks = load_hooks_from_file(hooks_file, hook_manager)
        console.print(f"\n[bold]Loaded {len(hooks)} hooks[/bold]")

    # Create orchestrator
    orchestrator = ResearchOrchestrator(
        settings=settings,
        output_dir=output_dir,
        skills_registry=skills_registry,
        subagent_manager=subagent_manager,
        hook_manager=hook_manager,
    )

    # Run research
    console.print("\n[bold green]Starting research...[/bold green]\n")

    try:
        result = asyncio.run(orchestrator.run_research(
            problem=problem,
            input_files=input_files if input_files else None,
            focus_areas=focus_areas,
        ))

        console.print(Panel.fit(
            f"[bold green]Research Complete![/bold green]\n\n"
            f"Session ID: {result['session_id']}\n"
            f"Output: {result['output_dir']}",
            title="Success",
        ))

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logger.exception("Research failed")
        raise typer.Exit(1) from e


@app.command()
def sessions(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status (active, completed)"),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option("--env", help="Path to .env file"),
    ] = None,
) -> None:
    """List all research sessions."""
    settings = get_settings(env_file)

    from .backends.persistence import SQLiteStore
    store = SQLiteStore(settings.sqlite_db_path)
    sessions_list = store.list_sessions(status)

    if not sessions_list:
        console.print("[yellow]No research sessions found.[/yellow]")
        return

    table = Table(title="Research Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Problem", max_width=40)
    table.add_column("Status", style="green")
    table.add_column("Created")

    for session in sessions_list:
        problem = session["problem"][:40] + "..." if len(session["problem"]) > 40 else session["problem"]
        table.add_row(
            session["id"],
            problem,
            session["status"],
            session["created_at"],
        )

    console.print(table)


@app.command()
def session(
    session_id: Annotated[str, typer.Argument(help="The session ID to view")],
    env_file: Annotated[
        Path | None,
        typer.Option("--env", help="Path to .env file"),
    ] = None,
) -> None:
    """View details of a specific research session."""
    settings = get_settings(env_file)

    from .backends.persistence import PersistenceManager
    persistence = PersistenceManager(
        db_path=settings.sqlite_db_path,
        base_output_dir=Path("./research"),
    )

    session_data = persistence.store.get_session(session_id)
    if not session_data:
        console.print(f"[red]Session not found:[/red] {session_id}")
        raise typer.Exit(1)

    agents = persistence.store.get_session_agents(session_id)

    console.print(Panel.fit(
        f"[bold]Session:[/bold] {session_id}\n"
        f"[bold]Status:[/bold] {session_data['status']}\n"
        f"[bold]Created:[/bold] {session_data['created_at']}\n\n"
        f"[bold]Problem:[/bold]\n{session_data['problem']}",
        title="Research Session",
    ))

    if agents:
        table = Table(title="Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Focus Area")
        table.add_column("Status", style="green")

        for agent in agents:
            table.add_row(
                agent["agent_name"],
                agent["focus_area"] or "-",
                agent["status"],
            )

        console.print(table)


@app.command()
def init(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory to initialize"),
    ] = Path(),
    with_skills: Annotated[
        bool,
        typer.Option("--skills/--no-skills", help="Create sample skills directory"),
    ] = True,
    with_subagents: Annotated[
        bool,
        typer.Option("--subagents/--no-subagents", help="Create sample subagents.jsonl"),
    ] = True,
    with_hooks: Annotated[
        bool,
        typer.Option("--hooks/--no-hooks", help="Create sample hooks.py"),
    ] = True,
) -> None:
    """Initialize a new Brainstormer project directory."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Initializing Brainstormer in {directory}[/bold]\n")

    # Copy .env.sample
    env_sample = directory / ".env.sample"
    if not env_sample.exists():
        try:
            # Try to read from package
            import brainstormer
            package_env = Path(brainstormer.__file__).parent.parent.parent / ".env.sample"
            if package_env.exists():
                env_sample.write_text(package_env.read_text())
        except Exception:
            # Create minimal .env.sample
            env_sample.write_text("""# LLM API Keys
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
""")
        console.print("  Created .env.sample")

    # Create skills directory
    if with_skills:
        skills_dir = create_skill_directory(directory)
        console.print(f"  Created skills directory: {skills_dir}")

    # Create subagents.jsonl
    if with_subagents:
        subagents_file = directory / "subagents.jsonl"
        if not subagents_file.exists():
            create_default_subagents_file(subagents_file)
            console.print("  Created subagents.jsonl")

    # Create sample hooks.py
    if with_hooks:
        hooks_file = directory / "hooks.py"
        if not hooks_file.exists():
            hooks_file.write_text('''"""Custom hooks for Brainstormer."""

from brainstormer.middleware.hooks import hook, HookPhase, HookResult


@hook("plan_creation", HookPhase.PRE, name="validate_plan")
def validate_plan(data: dict, context: dict) -> HookResult:
    """Validate plan before creation."""
    # Add custom validation logic here
    return HookResult(success=True)


@hook("search", HookPhase.POST, name="log_search")
def log_search(data: dict, context: dict) -> HookResult:
    """Log search queries and results."""
    query = data.get("query", {}).get("query", "")
    results_count = len(data.get("results", []))
    print(f"Search: {query} -> {results_count} results")
    return HookResult(success=True)


@hook("completion", HookPhase.POST, name="notify_completion")
async def notify_completion(data: dict, context: dict) -> HookResult:
    """Notify when an agent completes."""
    agent_name = data.get("completion_data", {}).get("agent_name", "unknown")
    print(f"Agent completed: {agent_name}")
    return HookResult(success=True)
''')
            console.print("  Created hooks.py")

    # Create research output directory
    research_dir = directory / "research"
    research_dir.mkdir(exist_ok=True)
    console.print("  Created research/ directory")

    console.print("\n[bold green]Initialization complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Copy .env.sample to .env and add your API keys")
    console.print("  2. Customize subagents.jsonl with your research agents")
    console.print("  3. Add skills to the skills/ directory")
    console.print("  4. Run: brainstormer research \"Your research question\"")


@app.command()
def memory(
    query: Annotated[
        str | None,
        typer.Argument(help="Search query for memories"),
    ] = None,
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Number of results to show"),
    ] = 10,
    env_file: Annotated[
        Path | None,
        typer.Option("--env", help="Path to .env file"),
    ] = None,
) -> None:
    """Search or list memories from long-term storage."""
    settings = get_settings(env_file)

    chroma_store = ChromaMemoryStore(persist_directory=settings.chromadb_path)
    memory_manager = MemoryManager(chroma_store)

    if query:
        console.print(f"[bold]Searching memories for:[/bold] {query}\n")
        memories = memory_manager.recall_relevant(query, n_results=count)
    else:
        console.print("[bold]Recent memories[/bold]\n")
        # Get recent memories with a broad query
        memories = memory_manager.recall_relevant("research insights findings", n_results=count)

    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        return

    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")[:200]
        metadata = mem.get("metadata", {})
        console.print(Panel(
            f"{content}{'...' if len(mem.get('content', '')) > 200 else ''}\n\n"
            f"[dim]Type: {metadata.get('type', 'unknown')} | "
            f"Session: {metadata.get('session_id', 'N/A')}[/dim]",
            title=f"Memory {i}",
        ))


@app.command()
def skills(
    skills_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Skills directory path"),
    ] = None,
) -> None:
    """List available skills."""
    if skills_dir is None:
        skills_dir = Path("./skills")

    if not skills_dir.exists():
        console.print(f"[yellow]Skills directory not found:[/yellow] {skills_dir}")
        console.print("Run 'brainstormer init' to create a sample skills directory.")
        return

    registry = SkillRegistry(skills_dir)
    skills_list = registry.list_all()

    if not skills_list:
        console.print("[yellow]No skills found.[/yellow]")
        return

    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for skill in skills_list:
        table.add_row(skill.name, skill.description[:60] + "..." if len(skill.description) > 60 else skill.description)

    console.print(table)


@app.command()
def subagents(
    subagents_file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Path to subagents.jsonl"),
    ] = None,
) -> None:
    """List configured subagents."""
    if subagents_file is None:
        subagents_file = Path("./subagents.jsonl")

    if not subagents_file.exists():
        console.print(f"[yellow]Subagents file not found:[/yellow] {subagents_file}")
        console.print("Run 'brainstormer init' to create a sample subagents.jsonl.")
        return

    manager = SubagentManager(subagents_file)
    agents = manager.list_all()

    if not agents:
        console.print("[yellow]No subagents configured.[/yellow]")
        return

    table = Table(title="Configured Subagents")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Focus Areas")

    for agent in agents:
        focus = ", ".join(agent.focus_areas[:3])
        if len(agent.focus_areas) > 3:
            focus += f" (+{len(agent.focus_areas) - 3})"
        table.add_row(
            agent.name,
            agent.description[:40] + "..." if len(agent.description) > 40 else agent.description,
            focus or "-",
        )

    console.print(table)


if __name__ == "__main__":
    app()

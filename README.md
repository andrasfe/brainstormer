# Brainstormer

A CLI for orchestrated research with configurable subagents, long-term memory, and extensible hooks.

Built on [LangChain DeepAgents](https://github.com/langchain-ai/deepagents), Brainstormer enables complex, multi-step research tasks through intelligent agent coordination.

## Features

- **Orchestrated Research**: Main agent creates plans and delegates to specialized subagents
- **Configurable Subagents**: Define agents via JSONL with custom prompts, tools, and focus areas
- **Long-term Memory**: SQLite for state persistence + ChromaDB for semantic search
- **Extensible Hooks**: Full lifecycle middleware for custom logic injection
- **Skills System**: Load domain-specific instructions following [Anthropic's skills format](https://github.com/anthropics/skills)
- **Multi-Provider LLM Support**: Works with Claude (Anthropic), GPT (OpenAI), and OpenRouter
- **Web Search**: Integrated Tavily search for real-time research
- **PDF/Text Parsing**: Include documents as research context

## Installation

```bash
pip install brainstormer
```

Or install from source:

```bash
git clone https://github.com/your-org/brainstormer.git
cd brainstormer
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize a project

```bash
brainstormer init my-research
cd my-research
```

This creates:
```
my-research/
├── .env.sample       # API key template
├── skills/           # Skill definitions
│   └── research-assistant/
│       └── SKILL.md
├── subagents.jsonl   # Subagent configurations
├── hooks.py          # Custom middleware hooks
└── research/         # Output directory
```

### 2. Configure API keys

```bash
cp .env.sample .env
# Edit .env with your API keys
```

Required keys:
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` or `OPENROUTER_API_KEY` - LLM provider
- `TAVILY_API_KEY` - Web search

### 3. Run research

```bash
brainstormer research "What are the latest advances in quantum computing?"
```

With input files:
```bash
brainstormer research "Analyze this market" \
  --file report.pdf \
  --file notes.txt \
  --focus "competitive landscape" \
  --focus "market trends"
```

## CLI Commands

### `brainstormer research`

Start a new research session.

```bash
brainstormer research "Your research question" [OPTIONS]

Options:
  -f, --file PATH      Input files (text/PDF) for context
  --focus TEXT         Predefined focus areas
  -o, --output PATH    Output directory (default: ./research)
  -s, --subagents PATH JSONL file with subagent configs
  --skills PATH        Skills directory
  --hooks PATH         Python file with hook definitions
  --env PATH           Path to .env file
  -v, --verbose        Enable verbose logging
```

### `brainstormer sessions`

List all research sessions.

```bash
brainstormer sessions [--status active|completed]
```

### `brainstormer session <id>`

View details of a specific session.

### `brainstormer memory`

Search long-term memory.

```bash
brainstormer memory "quantum computing"  # Search
brainstormer memory --count 20           # List recent
```

### `brainstormer skills`

List available skills.

### `brainstormer subagents`

List configured subagents.

### `brainstormer init`

Initialize a new project directory.

## Configuration

### Subagents (subagents.jsonl)

Define specialized research agents:

```jsonl
{"name": "market-analyst", "description": "Market research specialist", "system_prompt": "You are a market analyst...", "focus_areas": ["market", "competitive"], "tools": ["internet_search"]}
{"name": "tech-researcher", "description": "Technical research expert", "system_prompt": "You are a technical researcher...", "focus_areas": ["technical", "architecture"]}
```

Fields:
- `name`: Unique identifier
- `description`: What the agent does
- `system_prompt`: Agent instructions
- `focus_areas`: Keywords for automatic matching
- `tools`: Enabled tools (internet_search, write_file, etc.)
- `model`: Override default LLM model
- `max_depth`: Max recursion for sub-subagents

### Skills (skills/*/SKILL.md)

Skills follow [Anthropic's format](https://github.com/anthropics/skills):

```markdown
---
name: my-skill
description: What this skill does
---

# My Skill

Instructions for the agent when this skill is active.

## Guidelines
- Guideline 1
- Guideline 2
```

### Hooks (hooks.py)

Add custom middleware for lifecycle events:

```python
from brainstormer.middleware.hooks import hook, HookPhase, HookResult

@hook("plan_creation", HookPhase.PRE, name="validate_plan")
def validate_plan(data: dict, context: dict) -> HookResult:
    # Modify or validate plan before creation
    return HookResult(success=True, modified_data=data)

@hook("search", HookPhase.POST, name="log_search")
async def log_search(data: dict, context: dict) -> HookResult:
    print(f"Search: {data['query']} -> {len(data['results'])} results")
    return HookResult(success=True)

@hook("agent_spawn", HookPhase.PRE, name="approve_agent")
def approve_agent(data: dict, context: dict) -> HookResult:
    if some_condition:
        return HookResult(success=True, should_abort=True)
    return HookResult(success=True)
```

Available events:
- `session_start` / `session_end`
- `plan_creation`
- `agent_spawn`
- `research_write`
- `search`
- `completion`
- `memory_store` / `memory_recall`
- `skill_load`
- `tool_call`

### Environment Variables

```bash
# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# Web Search
TAVILY_API_KEY=tvly-...

# LLM Settings
DEFAULT_LLM_PROVIDER=anthropic  # or openai, openrouter
DEFAULT_LLM_MODEL=claude-sonnet-4-5-20250929
# For OpenRouter: DEFAULT_LLM_MODEL=anthropic/claude-3.5-sonnet

# Embeddings
EMBEDDING_PROVIDER=openai  # or local
EMBEDDING_MODEL=text-embedding-3-small

# Paths (optional)
SQLITE_DB_PATH=./brainstormer.db
CHROMADB_PATH=./chromadb
SKILLS_DIR=./skills

# Logging
LOG_LEVEL=INFO
```

## Output Structure

Research sessions create organized output:

```
research/
└── research-20241226-143022-abc123/
    ├── RESEARCH_PLAN.md          # Main plan and agent assignments
    ├── market-analyst/           # Agent output directory
    │   ├── findings.md
    │   └── competitive-analysis/
    │       └── report.md
    ├── tech-researcher/
    │   └── technical-review.md
    └── synthesis.md              # Final synthesized report
```

## Development

```bash
# Install dev dependencies
make install-dev

# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test

# Run tests with coverage
make test-cov

# Run all checks
make check

# Format code
make format
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (Typer)                          │
├─────────────────────────────────────────────────────────────┤
│                   Research Orchestrator                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Skills    │  │  Subagents  │  │   Middleware/Hooks  │  │
│  │  Registry   │  │   Manager   │  │      Manager        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    DeepAgents Framework                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Planning  │  │  Subagent   │  │    File System      │  │
│  │   (Todos)   │  │  Delegation │  │    Backend          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Persistence Layer                       │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │  SQLite (State)     │  │  ChromaDB (Vector Memory)   │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                       LLM Providers                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │   Anthropic   │  │    OpenAI     │  │  OpenRouter   │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT

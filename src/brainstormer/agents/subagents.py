"""Subagent configuration and management via JSONL."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""

    name: str
    description: str
    system_prompt: str
    focus_areas: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # Tool names to enable
    model: str | None = None  # Override default model
    max_depth: int = 2  # Max recursion depth for sub-subagents
    capabilities: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_deepagent_config(self) -> dict:
        """Convert to deepagents subagent format."""
        config = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
        }
        if self.model:
            config["model"] = self.model
        return config

    @classmethod
    def from_dict(cls, data: dict) -> "SubagentConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            focus_areas=data.get("focus_areas", []),
            tools=data.get("tools", []),
            model=data.get("model"),
            max_depth=data.get("max_depth", 2),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "focus_areas": self.focus_areas,
            "tools": self.tools,
            "model": self.model,
            "max_depth": self.max_depth,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }


def load_subagents_from_jsonl(file_path: Path) -> list[SubagentConfig]:
    """Load subagent configurations from a JSONL file."""
    subagents: list[SubagentConfig] = []

    if not file_path.exists():
        logger.warning(f"Subagents file not found: {file_path}")
        return subagents

    with file_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                data = json.loads(line)
                subagent = SubagentConfig.from_dict(data)
                subagents.append(subagent)
                logger.debug(f"Loaded subagent config: {subagent.name}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON on line {line_num}: {e}")
            except KeyError as e:
                logger.error(f"Missing required field on line {line_num}: {e}")

    logger.info(f"Loaded {len(subagents)} subagent configurations from {file_path}")
    return subagents


def save_subagents_to_jsonl(subagents: list[SubagentConfig], file_path: Path) -> None:
    """Save subagent configurations to a JSONL file."""
    with file_path.open("w", encoding="utf-8") as f:
        for subagent in subagents:
            f.write(json.dumps(subagent.to_dict()) + "\n")

    logger.info(f"Saved {len(subagents)} subagent configurations to {file_path}")


class SubagentManager:
    """Manages subagent configurations and instantiation."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path
        self._configs: dict[str, SubagentConfig] = {}

        if config_path and config_path.exists():
            self.reload()

    def reload(self) -> None:
        """Reload configurations from file."""
        if self.config_path:
            self._configs.clear()
            for config in load_subagents_from_jsonl(self.config_path):
                self._configs[config.name] = config

    def get(self, name: str) -> SubagentConfig | None:
        """Get a subagent configuration by name."""
        return self._configs.get(name)

    def list_all(self) -> list[SubagentConfig]:
        """List all configurations."""
        return list(self._configs.values())

    def register(self, config: SubagentConfig) -> None:
        """Register a configuration."""
        self._configs[config.name] = config

    def save(self) -> None:
        """Save current configurations to file."""
        if self.config_path:
            save_subagents_to_jsonl(list(self._configs.values()), self.config_path)

    def match_for_focus(self, focus_area: str) -> list[SubagentConfig]:
        """Find subagents suitable for a focus area."""
        matches = []
        focus_lower = focus_area.lower()

        for config in self._configs.values():
            # Check if focus area matches any configured focus areas
            if any(fa.lower() in focus_lower or focus_lower in fa.lower()
                   for fa in config.focus_areas) or focus_lower in config.description.lower() or any(cap.lower() in focus_lower for cap in config.capabilities):
                matches.append(config)

        return matches

    def create_dynamic_subagent(
        self,
        name: str,
        focus_area: str,
        base_prompt: str = "",
    ) -> SubagentConfig:
        """Create a dynamic subagent configuration for a focus area."""
        system_prompt = f"""{base_prompt}

## Focus Area: {focus_area}

You are a specialized research agent focused on: {focus_area}

Your responsibilities:
1. Conduct thorough research on this specific area
2. Search for relevant information using web search when needed
3. Analyze and synthesize findings
4. Write clear, structured output documenting your research
5. Create subdirectories for sub-topics if the research is complex

## Output Guidelines

- Write findings to markdown files in your assigned directory
- Use clear headings and structure
- Cite sources when applicable
- Highlight key insights and recommendations
"""

        return SubagentConfig(
            name=name,
            description=f"Research agent specialized in: {focus_area}",
            system_prompt=system_prompt,
            focus_areas=[focus_area],
            tools=["internet_search", "write_file", "read_file", "ls"],
            capabilities=["research", "analysis", "writing"],
        )


def create_default_subagents_file(file_path: Path) -> None:
    """Create a default subagents.jsonl file with example configurations."""
    default_subagents = [
        SubagentConfig(
            name="literature-researcher",
            description="Expert at finding and analyzing academic and technical literature",
            system_prompt="""You are an expert literature researcher. Your task is to:
1. Search for relevant academic papers, articles, and technical documentation
2. Analyze and summarize key findings
3. Identify trends and patterns in the literature
4. Note gaps in existing research
5. Compile comprehensive literature reviews

Focus on accuracy and proper attribution of sources.""",
            focus_areas=["literature", "academic", "papers", "research"],
            tools=["internet_search", "write_file", "read_file"],
            capabilities=["literature_review", "academic_research", "citation"],
        ),
        SubagentConfig(
            name="market-analyst",
            description="Specialist in market research, trends, and competitive analysis",
            system_prompt="""You are a market research analyst. Your responsibilities:
1. Research market trends and dynamics
2. Analyze competitive landscape
3. Identify market opportunities and threats
4. Gather data on pricing, positioning, and market share
5. Compile actionable market intelligence reports

Use multiple sources to validate findings.""",
            focus_areas=["market", "competitive", "trends", "business"],
            tools=["internet_search", "write_file", "read_file"],
            capabilities=["market_research", "competitive_analysis", "trend_analysis"],
        ),
        SubagentConfig(
            name="technical-analyst",
            description="Expert in technical research, architecture analysis, and technology evaluation",
            system_prompt="""You are a technical research analyst. Your focus:
1. Research technical implementations and architectures
2. Evaluate technologies and frameworks
3. Analyze technical trade-offs
4. Document best practices and patterns
5. Identify technical risks and considerations

Provide detailed technical analysis with practical recommendations.""",
            focus_areas=["technical", "technology", "architecture", "engineering"],
            tools=["internet_search", "write_file", "read_file"],
            capabilities=["technical_research", "architecture_analysis", "tech_evaluation"],
        ),
        SubagentConfig(
            name="data-researcher",
            description="Specialist in finding, analyzing, and synthesizing data from various sources",
            system_prompt="""You are a data research specialist. Your tasks:
1. Find relevant data sources and datasets
2. Analyze quantitative information
3. Identify statistical trends and patterns
4. Validate data quality and reliability
5. Present data-driven insights clearly

Focus on accuracy and proper data interpretation.""",
            focus_areas=["data", "statistics", "quantitative", "metrics"],
            tools=["internet_search", "write_file", "read_file"],
            capabilities=["data_analysis", "statistical_research", "quantitative_analysis"],
        ),
    ]

    save_subagents_to_jsonl(default_subagents, file_path)
    logger.info(f"Created default subagents file at {file_path}")

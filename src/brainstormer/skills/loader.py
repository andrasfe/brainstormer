"""Skills loader following Anthropic's skills format."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Skill:
    """A loaded skill with instructions and metadata."""

    name: str
    description: str
    instructions: str
    path: Path
    metadata: dict = field(default_factory=dict)

    def to_system_prompt(self) -> str:
        """Convert skill to system prompt format."""
        return f"""## Skill: {self.name}

{self.description}

### Instructions

{self.instructions}
"""


class SkillLoader:
    """Loads skills from SKILL.md files following Anthropic's format."""

    SKILL_FILENAME = "SKILL.md"
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir

    def load_skill(self, skill_path: Path) -> Skill | None:
        """Load a single skill from a directory or SKILL.md file."""
        skill_file = skill_path / self.SKILL_FILENAME if skill_path.is_dir() else skill_path

        if not skill_file.exists():
            logger.warning(f"Skill file not found: {skill_file}")
            return None

        content = skill_file.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter_match = self.FRONTMATTER_PATTERN.match(content)
        if not frontmatter_match:
            logger.warning(f"No frontmatter found in {skill_file}")
            return None

        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML frontmatter in {skill_file}: {e}")
            return None

        if not isinstance(frontmatter, dict):
            logger.error(f"Frontmatter must be a dict in {skill_file}")
            return None

        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name or not description:
            logger.error(f"Missing required fields (name, description) in {skill_file}")
            return None

        # Extract instructions (content after frontmatter)
        instructions = content[frontmatter_match.end():].strip()

        # Extract additional metadata
        metadata = {k: v for k, v in frontmatter.items() if k not in {"name", "description"}}

        skill = Skill(
            name=name,
            description=description,
            instructions=instructions,
            path=skill_file.parent if skill_file.name == self.SKILL_FILENAME else skill_file,
            metadata=metadata,
        )

        logger.info(f"Loaded skill: {name}")
        return skill

    def load_all(self) -> list[Skill]:
        """Load all skills from the skills directory."""
        skills: list[Skill] = []

        if not self.skills_dir.exists():
            logger.info(f"Skills directory does not exist: {self.skills_dir}")
            return skills

        # Look for SKILL.md in subdirectories
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill = self.load_skill(skill_dir)
                if skill:
                    skills.append(skill)

        # Also check for SKILL.md files directly in skills_dir
        for skill_file in self.skills_dir.glob("*.md"):
            if skill_file.name != self.SKILL_FILENAME:
                # Treat .md files with frontmatter as skills
                skill = self.load_skill(skill_file)
                if skill:
                    skills.append(skill)

        logger.info(f"Loaded {len(skills)} skills from {self.skills_dir}")
        return skills


class SkillRegistry:
    """Registry for managing loaded skills."""

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir
        self._skills: dict[str, Skill] = {}
        self._loader: SkillLoader | None = None

        if skills_dir:
            self._loader = SkillLoader(skills_dir)
            self.reload()

    def reload(self) -> None:
        """Reload all skills from the skills directory."""
        if self._loader:
            self._skills.clear()
            for skill in self._loader.load_all():
                self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def register(self, skill: Skill) -> None:
        """Register a skill manually."""
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a skill by name."""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get_combined_prompt(self, skill_names: list[str] | None = None) -> str:
        """Get combined system prompt for selected skills."""
        if skill_names:
            skills = [self._skills[name] for name in skill_names if name in self._skills]
        else:
            skills = list(self._skills.values())

        if not skills:
            return ""

        prompts = [skill.to_system_prompt() for skill in skills]
        return "\n\n---\n\n".join(prompts)

    def match_skills(self, query: str) -> list[Skill]:
        """Find skills relevant to a query based on description."""
        query_lower = query.lower()
        matches = []

        for skill in self._skills.values():
            # Simple keyword matching
            if any(
                word in skill.description.lower() or word in skill.name.lower()
                for word in query_lower.split()
            ):
                matches.append(skill)

        return matches


def create_skill_directory(base_dir: Path) -> Path:
    """Create the skills directory structure."""
    skills_dir = base_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample skill
    sample_skill_dir = skills_dir / "research-assistant"
    sample_skill_dir.mkdir(exist_ok=True)

    sample_skill = sample_skill_dir / "SKILL.md"
    if not sample_skill.exists():
        sample_skill.write_text("""---
name: research-assistant
description: Expert at conducting thorough research and synthesizing findings into clear reports
---

# Research Assistant

You are an expert research assistant skilled at:

- Breaking down complex research questions into manageable components
- Conducting systematic literature and web searches
- Synthesizing information from multiple sources
- Identifying key insights and patterns
- Writing clear, well-structured research reports

## Guidelines

1. Always start by understanding the core research question
2. Create a systematic search strategy before diving into research
3. Keep track of all sources and cite them properly
4. Look for conflicting information and address discrepancies
5. Synthesize findings into actionable insights
6. Present information in a clear, logical structure

## Output Format

Structure your research outputs as:

1. **Executive Summary** - Key findings in 2-3 paragraphs
2. **Methodology** - How the research was conducted
3. **Detailed Findings** - Organized by topic/theme
4. **Sources** - All references used
5. **Recommendations** - Actionable next steps
""", encoding="utf-8")

    logger.info(f"Created skills directory at {skills_dir}")
    return skills_dir

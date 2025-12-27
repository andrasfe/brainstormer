"""Tests for skills loader."""

from pathlib import Path

from brainstormer.skills.loader import (
    Skill,
    SkillLoader,
    SkillRegistry,
    create_skill_directory,
)


class TestSkill:
    """Tests for Skill dataclass."""

    def test_skill_creation(self):
        """Test creating a skill."""
        skill = Skill(
            name="test-skill",
            description="A test skill",
            instructions="Do something",
            path=Path("/tmp/skill"),
        )

        assert skill.name == "test-skill"
        assert skill.description == "A test skill"

    def test_skill_to_system_prompt(self):
        """Test converting skill to system prompt."""
        skill = Skill(
            name="test-skill",
            description="A test skill",
            instructions="Do something\n\nMore instructions",
            path=Path("/tmp/skill"),
        )

        prompt = skill.to_system_prompt()

        assert "## Skill: test-skill" in prompt
        assert "A test skill" in prompt
        assert "Do something" in prompt


class TestSkillLoader:
    """Tests for SkillLoader."""

    def test_load_skill(self, sample_skill_dir):
        """Test loading a single skill."""
        loader = SkillLoader(sample_skill_dir)
        skill = loader.load_skill(sample_skill_dir / "test-skill")

        assert skill is not None
        assert skill.name == "test-skill"
        assert "test skill for unit testing" in skill.description

    def test_load_skill_no_frontmatter(self, temp_dir):
        """Test loading skill without frontmatter fails gracefully."""
        skill_dir = temp_dir / "bad-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# No frontmatter here")

        loader = SkillLoader(temp_dir)
        skill = loader.load_skill(skill_dir)

        assert skill is None

    def test_load_all_skills(self, sample_skill_dir):
        """Test loading all skills from directory."""
        loader = SkillLoader(sample_skill_dir)
        skills = loader.load_all()

        assert len(skills) >= 1
        assert any(s.name == "test-skill" for s in skills)


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_registry_initialization(self, sample_skill_dir):
        """Test registry loads skills on init."""
        registry = SkillRegistry(sample_skill_dir)

        assert len(registry.list_all()) >= 1

    def test_get_skill(self, sample_skill_dir):
        """Test getting a skill by name."""
        registry = SkillRegistry(sample_skill_dir)

        skill = registry.get("test-skill")
        assert skill is not None
        assert skill.name == "test-skill"

    def test_get_nonexistent_skill(self, sample_skill_dir):
        """Test getting a non-existent skill."""
        registry = SkillRegistry(sample_skill_dir)

        skill = registry.get("nonexistent")
        assert skill is None

    def test_register_skill(self):
        """Test manually registering a skill."""
        registry = SkillRegistry()
        skill = Skill(
            name="manual-skill",
            description="Manually added",
            instructions="Instructions",
            path=Path("/tmp"),
        )

        registry.register(skill)

        assert registry.get("manual-skill") is not None

    def test_unregister_skill(self, sample_skill_dir):
        """Test unregistering a skill."""
        registry = SkillRegistry(sample_skill_dir)

        result = registry.unregister("test-skill")
        assert result is True
        assert registry.get("test-skill") is None

    def test_get_combined_prompt(self, sample_skill_dir):
        """Test getting combined prompt for all skills."""
        registry = SkillRegistry(sample_skill_dir)

        prompt = registry.get_combined_prompt()

        assert "## Skill:" in prompt

    def test_match_skills(self, sample_skill_dir):
        """Test matching skills by query."""
        registry = SkillRegistry(sample_skill_dir)

        matches = registry.match_skills("test")
        assert len(matches) >= 1


class TestCreateSkillDirectory:
    """Tests for skill directory creation."""

    def test_create_skill_directory(self, temp_dir):
        """Test creating skills directory with sample."""
        skills_dir = create_skill_directory(temp_dir)

        assert skills_dir.exists()
        assert (skills_dir / "research-assistant" / "SKILL.md").exists()

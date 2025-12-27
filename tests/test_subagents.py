"""Tests for subagent configuration and management."""



from brainstormer.agents.subagents import (
    SubagentConfig,
    SubagentManager,
    create_default_subagents_file,
    load_subagents_from_jsonl,
    save_subagents_to_jsonl,
)


class TestSubagentConfig:
    """Tests for SubagentConfig dataclass."""

    def test_config_creation(self):
        """Test creating a subagent config."""
        config = SubagentConfig(
            name="test-agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            focus_areas=["testing", "qa"],
        )

        assert config.name == "test-agent"
        assert "testing" in config.focus_areas
        assert config.max_depth == 2  # default

    def test_to_deepagent_config(self):
        """Test converting to deepagent format."""
        config = SubagentConfig(
            name="test-agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            model="gpt-4o",
        )

        deep_config = config.to_deepagent_config()

        assert deep_config["name"] == "test-agent"
        assert deep_config["description"] == "A test agent"
        assert deep_config["model"] == "gpt-4o"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "from-dict",
            "description": "Created from dict",
            "system_prompt": "Prompt here",
            "focus_areas": ["area1", "area2"],
            "max_depth": 3,
        }

        config = SubagentConfig.from_dict(data)

        assert config.name == "from-dict"
        assert config.max_depth == 3
        assert len(config.focus_areas) == 2

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = SubagentConfig(
            name="test",
            description="test",
            system_prompt="prompt",
            focus_areas=["a", "b"],
        )

        data = config.to_dict()

        assert data["name"] == "test"
        assert "focus_areas" in data


class TestLoadSubagents:
    """Tests for JSONL loading."""

    def test_load_subagents(self, sample_subagents_file):
        """Test loading subagents from JSONL."""
        configs = load_subagents_from_jsonl(sample_subagents_file)

        assert len(configs) == 2
        assert configs[0].name == "test-agent-1"
        assert configs[1].name == "test-agent-2"

    def test_load_empty_file(self, temp_dir):
        """Test loading from empty file."""
        file_path = temp_dir / "empty.jsonl"
        file_path.write_text("")

        configs = load_subagents_from_jsonl(file_path)
        assert len(configs) == 0

    def test_load_with_comments(self, temp_dir):
        """Test loading with comment lines."""
        file_path = temp_dir / "comments.jsonl"
        file_path.write_text("""\
# This is a comment
{"name": "agent", "description": "desc", "system_prompt": "prompt"}
# Another comment
""")

        configs = load_subagents_from_jsonl(file_path)
        assert len(configs) == 1

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading from non-existent file."""
        configs = load_subagents_from_jsonl(temp_dir / "nonexistent.jsonl")
        assert len(configs) == 0


class TestSaveSubagents:
    """Tests for JSONL saving."""

    def test_save_subagents(self, temp_dir):
        """Test saving subagents to JSONL."""
        configs = [
            SubagentConfig(
                name="agent1",
                description="Agent 1",
                system_prompt="Prompt 1",
            ),
            SubagentConfig(
                name="agent2",
                description="Agent 2",
                system_prompt="Prompt 2",
            ),
        ]

        file_path = temp_dir / "output.jsonl"
        save_subagents_to_jsonl(configs, file_path)

        # Reload and verify
        loaded = load_subagents_from_jsonl(file_path)
        assert len(loaded) == 2
        assert loaded[0].name == "agent1"


class TestSubagentManager:
    """Tests for SubagentManager."""

    def test_manager_initialization(self, sample_subagents_file):
        """Test manager loads configs on init."""
        manager = SubagentManager(sample_subagents_file)

        assert len(manager.list_all()) == 2

    def test_get_subagent(self, sample_subagents_file):
        """Test getting subagent by name."""
        manager = SubagentManager(sample_subagents_file)

        config = manager.get("test-agent-1")
        assert config is not None
        assert config.name == "test-agent-1"

    def test_match_for_focus(self, sample_subagents_file):
        """Test matching subagents for focus area."""
        manager = SubagentManager(sample_subagents_file)

        matches = manager.match_for_focus("testing")
        assert len(matches) >= 1
        assert any(m.name == "test-agent-1" for m in matches)

    def test_create_dynamic_subagent(self, sample_subagents_file):
        """Test creating dynamic subagent config."""
        manager = SubagentManager(sample_subagents_file)

        config = manager.create_dynamic_subagent(
            name="dynamic-agent",
            focus_area="Machine Learning",
            base_prompt="Additional context here.",
        )

        assert config.name == "dynamic-agent"
        assert "Machine Learning" in config.system_prompt
        assert "Additional context" in config.system_prompt


class TestCreateDefaultSubagents:
    """Tests for default subagents creation."""

    def test_create_default_file(self, temp_dir):
        """Test creating default subagents file."""
        file_path = temp_dir / "subagents.jsonl"
        create_default_subagents_file(file_path)

        assert file_path.exists()

        configs = load_subagents_from_jsonl(file_path)
        assert len(configs) >= 4  # Default has 4 agents

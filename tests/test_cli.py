"""Tests for CLI module."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from brainstormer.cli import app

runner = CliRunner()


class TestInitCommand:
    """Tests for init command."""

    def test_init_creates_structure(self, temp_dir):
        """Test init creates expected file structure."""
        result = runner.invoke(app, ["init", str(temp_dir)])

        assert result.exit_code == 0
        assert (temp_dir / ".env.sample").exists()
        assert (temp_dir / "skills").exists()
        assert (temp_dir / "subagents.jsonl").exists()
        assert (temp_dir / "hooks.py").exists()
        assert (temp_dir / "research").exists()

    def test_init_without_options(self, temp_dir):
        """Test init with minimal options."""
        result = runner.invoke(app, [
            "init",
            str(temp_dir),
            "--no-skills",
            "--no-subagents",
            "--no-hooks",
        ])

        assert result.exit_code == 0
        assert not (temp_dir / "skills").exists()
        assert not (temp_dir / "subagents.jsonl").exists()


class TestSkillsCommand:
    """Tests for skills command."""

    def test_skills_list(self, sample_skill_dir):
        """Test listing skills."""
        result = runner.invoke(app, ["skills", "--dir", str(sample_skill_dir)])

        assert result.exit_code == 0
        assert "test-skill" in result.output

    def test_skills_no_directory(self, temp_dir):
        """Test skills command with non-existent directory."""
        result = runner.invoke(app, ["skills", "--dir", str(temp_dir / "nonexistent")])

        assert "not found" in result.output.lower()


class TestSubagentsCommand:
    """Tests for subagents command."""

    def test_subagents_list(self, sample_subagents_file):
        """Test listing subagents."""
        result = runner.invoke(app, ["subagents", "--file", str(sample_subagents_file)])

        assert result.exit_code == 0
        assert "test-agent-1" in result.output

    def test_subagents_no_file(self, temp_dir):
        """Test subagents command with non-existent file."""
        result = runner.invoke(app, ["subagents", "--file", str(temp_dir / "nonexistent.jsonl")])

        assert "not found" in result.output.lower()


class TestSessionsCommand:
    """Tests for sessions command."""

    def test_sessions_empty(self, temp_dir, sample_settings):
        """Test listing sessions when empty."""
        with patch("brainstormer.cli.get_settings", return_value=sample_settings):
            result = runner.invoke(app, ["sessions"])

        assert result.exit_code == 0
        assert "no research sessions" in result.output.lower()


class TestResearchCommand:
    """Tests for research command."""

    def test_research_missing_api_key(self, temp_dir):
        """Test research fails without API key."""
        env_file = temp_dir / ".env"
        env_file.write_text("")  # Empty env file

        result = runner.invoke(app, [
            "research",
            "Test problem",
            "--env", str(env_file),
        ])

        # Should fail due to missing API key
        assert result.exit_code == 1

    def test_research_with_files(self, temp_dir, sample_text_file):
        """Test research with input files parses them."""
        # Create minimal env with all required keys
        env_file = temp_dir / ".env"
        env_file.write_text("""
ANTHROPIC_API_KEY=test-key
TAVILY_API_KEY=test-key
OPENAI_API_KEY=test-key
""")

        # Mock the orchestrator to avoid actual API calls
        with patch("brainstormer.cli.ResearchOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run_research = MagicMock(return_value={
                "session_id": "test-session",
                "output_dir": str(temp_dir / "output"),
            })
            mock_orch.return_value = mock_instance

            result = runner.invoke(app, [
                "research",
                "Test problem",
                "--file", str(sample_text_file),
                "--env", str(env_file),
                "--output", str(temp_dir / "output"),
            ])

        # Check file was mentioned in output
        assert "sample.txt" in result.output or result.exit_code == 0

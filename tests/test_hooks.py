"""Tests for hooks system."""


import pytest

from brainstormer.middleware.hooks import (
    Hook,
    HookManager,
    HookPhase,
    HookResult,
    hook,
    load_hooks_from_file,
)


class TestHook:
    """Tests for Hook dataclass."""

    def test_hook_creation(self):
        """Test creating a hook."""
        h = Hook(
            name="test_hook",
            event="plan_creation",
            phase=HookPhase.PRE,
            handler=lambda data, ctx: data,
        )

        assert h.name == "test_hook"
        assert h.event == "plan_creation"
        assert h.enabled is True

    def test_hook_invalid_handler(self):
        """Test that non-callable handler raises error."""
        with pytest.raises(ValueError, match="must be callable"):
            Hook(
                name="test",
                event="plan_creation",
                phase=HookPhase.PRE,
                handler="not a function",
            )


class TestHookResult:
    """Tests for HookResult dataclass."""

    def test_success_result(self):
        """Test successful hook result."""
        result = HookResult(success=True, modified_data={"key": "value"})

        assert result.success is True
        assert result.modified_data == {"key": "value"}
        assert result.should_abort is False

    def test_abort_result(self):
        """Test abort hook result."""
        result = HookResult(success=True, should_abort=True)

        assert result.should_abort is True


class TestHookManager:
    """Tests for HookManager."""

    def test_register_hook(self):
        """Test registering a hook."""
        manager = HookManager()

        h = manager.register(
            event="plan_creation",
            handler=lambda data, ctx: data,
            name="my_hook",
        )

        assert h.name == "my_hook"
        assert len(manager.get_hooks("plan_creation")) == 1

    def test_register_invalid_event(self):
        """Test registering hook for invalid event."""
        manager = HookManager()

        with pytest.raises(ValueError, match="Unknown event"):
            manager.register(
                event="invalid_event",
                handler=lambda data, ctx: data,
            )

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        manager = HookManager()

        h = manager.register(
            event="plan_creation",
            handler=lambda data, ctx: data,
        )

        result = manager.unregister(h)

        assert result is True
        assert len(manager.get_hooks("plan_creation")) == 0

    @pytest.mark.asyncio
    async def test_execute_hooks(self):
        """Test executing hooks."""
        manager = HookManager()

        def modify_data(data, ctx):
            data["modified"] = True
            return data

        manager.register(
            event="plan_creation",
            handler=modify_data,
            phase=HookPhase.PRE,
        )

        data, results = await manager.execute_pre(
            "plan_creation",
            {"original": True},
        )

        assert data["modified"] is True
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_async_hook(self):
        """Test executing async hooks."""
        manager = HookManager()

        async def async_handler(data, ctx):
            data["async"] = True
            return HookResult(success=True, modified_data=data)

        manager.register(
            event="search",
            handler=async_handler,
            phase=HookPhase.POST,
        )

        data, _results = await manager.execute_post(
            "search",
            {"query": "test"},
        )

        assert data["async"] is True

    @pytest.mark.asyncio
    async def test_hook_priority(self):
        """Test hooks execute in priority order."""
        manager = HookManager()
        execution_order = []

        def hook1(data, ctx):
            execution_order.append(1)
            return data

        def hook2(data, ctx):
            execution_order.append(2)
            return data

        manager.register(
            event="plan_creation",
            handler=hook2,
            priority=10,
        )
        manager.register(
            event="plan_creation",
            handler=hook1,
            priority=1,
        )

        await manager.execute_pre("plan_creation", {})

        assert execution_order == [1, 2]

    @pytest.mark.asyncio
    async def test_hook_abort(self):
        """Test hook can abort execution."""
        manager = HookManager()
        executed = []

        def abort_hook(data, ctx):
            executed.append("abort")
            return HookResult(success=True, should_abort=True)

        def second_hook(data, ctx):
            executed.append("second")
            return data

        manager.register(
            event="agent_spawn",
            handler=abort_hook,
            priority=1,
        )
        manager.register(
            event="agent_spawn",
            handler=second_hook,
            priority=2,
        )

        await manager.execute_pre("agent_spawn", {})

        assert executed == ["abort"]

    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """Test hooks handle errors gracefully."""
        manager = HookManager()

        def failing_hook(data, ctx):
            raise ValueError("Test error")

        manager.register(
            event="completion",
            handler=failing_hook,
        )

        _data, results = await manager.execute_pre("completion", {"key": "value"})

        assert len(results) == 1
        assert results[0].success is False
        assert "Test error" in results[0].error


class TestHookDecorator:
    """Tests for hook decorator."""

    def test_hook_decorator(self):
        """Test hook decorator marks function."""

        @hook("plan_creation", HookPhase.PRE, name="decorated_hook")
        def my_hook(data, ctx):
            return data

        assert hasattr(my_hook, "_hook_config")
        assert my_hook._hook_config["event"] == "plan_creation"
        assert my_hook._hook_config["name"] == "decorated_hook"


class TestLoadHooksFromFile:
    """Tests for loading hooks from files."""

    def test_load_hooks_from_file(self, temp_dir):
        """Test loading hooks from a Python file."""
        hooks_file = temp_dir / "test_hooks.py"
        hooks_file.write_text('''
from brainstormer.middleware.hooks import hook, HookPhase, HookResult

@hook("plan_creation", HookPhase.PRE, name="file_hook")
def my_hook(data, ctx):
    return HookResult(success=True)
''')

        manager = HookManager()
        loaded = load_hooks_from_file(hooks_file, manager)

        assert len(loaded) == 1
        assert loaded[0].name == "file_hook"

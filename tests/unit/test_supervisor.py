"""Tests for process supervisor helpers."""

from pathlib import Path
from unittest.mock import patch

from cementic.supervisor import (
    ManagedProcess,
    force_kill,
    is_managed_process_alive,
    is_pid_running,
    load_supervisor_state,
    managed_process_pid,
    managed_process_start_token,
    process_start_token,
    save_supervisor_state,
    spawn_detached,
    wait_for_exit,
)


class TestIsPidRunning:
    """Tests for is_pid_running."""

    def test_running_pid(self) -> None:
        # os.kill(pid, 0) succeeds and the process is in a live state
        with patch("os.kill", return_value=None):
            with patch("cementic.supervisor._proc_stat_fields", return_value=["S", "1", "1"]):
                assert is_pid_running(1234) is True

    @patch("os.kill", side_effect=ProcessLookupError)
    def test_missing_pid(self, mock_kill) -> None:
        assert is_pid_running(1234) is False

    @patch("os.kill", side_effect=OSError)
    def test_os_error_pid(self, mock_kill) -> None:
        assert is_pid_running(1234) is False

    def test_zombie_pid_is_not_running(self) -> None:
        """A detached worker that exited but wasn't reaped still answers
        kill(pid, 0); treating it as running makes `cementic stop` wait out its
        full timeout and then wrongly tell the user to --force it."""
        with patch("os.kill", return_value=None):
            with patch("cementic.supervisor._proc_stat_fields", return_value=["Z", "1", "1"]):
                assert is_pid_running(1234) is False

    def test_without_proc_falls_back_to_signal_probe(self) -> None:
        """Non-Linux platforms have no /proc: keep the kill(pid, 0) answer."""
        with patch("os.kill", return_value=None):
            with patch("cementic.supervisor._proc_stat_fields", return_value=None):
                assert is_pid_running(1234) is True


class TestLoadSupervisorState:
    """Tests for load_supervisor_state."""

    def test_nonexistent_file(self, temp_dir: Path) -> None:
        path = temp_dir / "nonexistent.json"
        result = load_supervisor_state(path)
        assert result == {}

    def test_valid_json(self, temp_dir: Path) -> None:
        path = temp_dir / "state.json"
        path.write_text('{"processes": [{"pid": 1}], "collection": "c1"}')
        result = load_supervisor_state(path)
        assert result == {"processes": [{"pid": 1}], "collection": "c1"}

    def test_corrupted_json(self, temp_dir: Path) -> None:
        path = temp_dir / "state.json"
        path.write_text("not valid json at all")
        result = load_supervisor_state(path)
        assert result == {}


class TestSaveSupervisorState:
    """Tests for save_supervisor_state."""

    def test_save_and_load_roundtrip(self, temp_dir: Path) -> None:
        path = temp_dir / "state.json"
        data = {"processes": [{"name": "sw", "pid": 42}]}
        save_supervisor_state(path, data)
        loaded = load_supervisor_state(path)
        assert loaded == data

    def test_creates_parent_directories(self, temp_dir: Path) -> None:
        path = temp_dir / "deep" / "nested" / "state.json"
        data = {"key": "value"}
        save_supervisor_state(path, data)
        assert path.exists()
        assert path.parent.exists()


class TestSpawnDetached:
    """Tests for spawn_detached."""

    @patch("cementic.supervisor.subprocess.Popen")
    def test_spawns_process(self, mock_popen, temp_dir: Path) -> None:
        mock_popen.return_value.pid = 9999
        log_file = temp_dir / "worker.log"
        pid = spawn_detached(["echo", "hello"], log_file)
        assert pid == 9999
        mock_popen.assert_called_once()
        assert log_file.exists()

    @patch("cementic.supervisor.subprocess.Popen")
    def test_passes_start_new_session(self, mock_popen, temp_dir: Path) -> None:
        mock_popen.return_value.pid = 8888
        log_file = temp_dir / "worker.log"
        spawn_detached(["cmd"], log_file)
        call_kwargs = mock_popen.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("start_new_session") is True


class TestWaitForExit:
    """Tests for wait_for_exit."""

    def test_empty_list(self) -> None:
        assert wait_for_exit([]) == []

    @patch("cementic.supervisor.is_pid_running", return_value=False)
    def test_none_running_returns_empty(self, mock_running) -> None:
        assert wait_for_exit([1, 2, 3]) == []

    @patch("cementic.supervisor.is_pid_running", return_value=True)
    def test_all_running_times_out(self, mock_running) -> None:
        # Short timeout to force timeout path
        result = wait_for_exit([1, 2], timeout_seconds=0.01)
        assert len(result) == 2

    @patch("cementic.supervisor.is_pid_running", side_effect=[True, False, False])
    def test_some_exit_during_wait(self, mock_running) -> None:
        # First check: pid 1 running, pid 2 not → remaining [1]
        # Second check: pid 1 not running → []
        result = wait_for_exit([1, 2], timeout_seconds=5.0)
        assert result == []


class TestForceKill:
    """Tests for force_kill."""

    @patch("os.kill")
    @patch("cementic.supervisor.is_pid_running", return_value=False)
    def test_kill_succeeds_pid_dies(self, mock_running, mock_kill) -> None:
        result = force_kill([1234])
        assert result == []
        mock_kill.assert_called_once_with(1234, 9)

    @patch("os.kill", side_effect=ProcessLookupError)
    @patch("cementic.supervisor.is_pid_running")
    def test_kill_process_lookup_error(self, mock_running, mock_kill) -> None:
        # Already gone: nothing left to report, no liveness check needed.
        result = force_kill([1234])
        assert result == []
        mock_running.assert_not_called()

    @patch("os.kill", side_effect=PermissionError)
    def test_kill_not_permitted_is_reported_not_swallowed(self, mock_kill) -> None:
        # EPERM means the process exists but isn't ours to kill -- reporting it
        # as successfully killed would be a lie to the user.
        result = force_kill([1234])
        assert result == [1234]

    @patch("os.kill")
    @patch("cementic.supervisor.is_pid_running", return_value=True)
    def test_kill_succeeds_but_pid_survives(self, mock_running, mock_kill) -> None:
        result = force_kill([1234])
        assert result == [1234]

    def test_multiple_pids(self) -> None:
        # pid 1: kill OK, is_pid_running=False → removed
        # pid 2: kill raises ProcessLookupError → silently removed
        # pid 3: kill OK, is_pid_running=True → stays
        with patch("os.kill", side_effect=[None, ProcessLookupError(), None]):
            with patch(
                "cementic.supervisor.is_pid_running", side_effect=[False, True]
            ):
                result = force_kill([1, 2, 3])
                assert result == [3]


class TestManagedProcess:
    """Tests for ManagedProcess dataclass."""

    def test_creation(self) -> None:
        proc = ManagedProcess(name="sw", pid=42, log_file="/tmp/sw.log")
        assert proc.name == "sw"
        assert proc.pid == 42
        assert proc.log_file == "/tmp/sw.log"
        assert proc.start_token is None

    def test_start_token_field(self) -> None:
        proc = ManagedProcess(name="sw", pid=42, log_file="/tmp/sw.log", start_token="999")
        assert proc.start_token == "999"


class TestProcessStartToken:
    """Tests for process_start_token (/proc-based start-time read)."""

    def test_returns_token_for_self(self) -> None:
        import os

        # This process is alive, so its own start token must be readable.
        assert process_start_token(os.getpid()) is not None

    def test_none_when_unreadable(self) -> None:
        with patch("cementic.supervisor.Path.read_text", side_effect=OSError):
            assert process_start_token(1234) is None

    def test_parses_field_22_past_comm_with_spaces(self) -> None:
        # comm field contains spaces and a ')'. After the final ')', index 0 is
        # field 3 (state); starttime is field 22 => index 19. Put 4242 there.
        after_comm = [str(i) for i in range(19)] + ["4242"]
        stat = "1234 (weird )name) " + " ".join(after_comm)
        with patch("cementic.supervisor.Path.read_text", return_value=stat):
            assert process_start_token(1234) == "4242"


class TestManagedProcessPid:
    """Tests for managed_process_pid."""

    def test_returns_int_pid(self) -> None:
        assert managed_process_pid({"pid": 42}) == 42

    def test_missing_key_defaults_zero(self) -> None:
        assert managed_process_pid({}) == 0

    def test_non_int_defaults_zero(self) -> None:
        assert managed_process_pid({"pid": "not-int"}) == 0


class TestManagedProcessStartToken:
    """Tests for managed_process_start_token."""

    def test_returns_str_token(self) -> None:
        assert managed_process_start_token({"start_token": "999"}) == "999"

    def test_missing_key_returns_none(self) -> None:
        assert managed_process_start_token({}) is None

    def test_non_str_returns_none(self) -> None:
        assert managed_process_start_token({"start_token": 999}) is None


class TestIsManagedProcessAlive:
    """Tests for is_managed_process_alive (PID + start-token identity)."""

    def test_false_for_nonpositive_pid(self) -> None:
        assert is_managed_process_alive(0, "1") is False

    @patch("cementic.supervisor.is_pid_running", return_value=False)
    def test_false_when_not_running(self, _mock) -> None:
        assert is_managed_process_alive(1234, "1") is False

    @patch("cementic.supervisor.is_pid_running", return_value=True)
    def test_true_when_token_unknown(self, _mock) -> None:
        # Backward compat: no recorded token falls back to a PID-only check.
        assert is_managed_process_alive(1234, None) is True

    @patch("cementic.supervisor.process_start_token", return_value="111")
    @patch("cementic.supervisor.is_pid_running", return_value=True)
    def test_true_on_token_match(self, _run, _tok) -> None:
        assert is_managed_process_alive(1234, "111") is True

    @patch("cementic.supervisor.process_start_token", return_value="222")
    @patch("cementic.supervisor.is_pid_running", return_value=True)
    def test_false_on_token_mismatch_recycled_pid(self, _run, _tok) -> None:
        assert is_managed_process_alive(1234, "111") is False

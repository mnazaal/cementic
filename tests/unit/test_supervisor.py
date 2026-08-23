"""Tests for process supervisor helpers."""

from pathlib import Path
from unittest.mock import patch

from cementic.supervisor import (
    ManagedProcess,
    find_pids_by_cmdline,
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

    @patch("os.kill", side_effect=PermissionError)
    def test_permission_error_means_alive_not_gone(self, mock_kill) -> None:
        """EPERM proves the process exists -- it is just not ours to signal.

        Reporting it as gone made `cementic stop` print success and clear the
        state files while workers started under another uid kept running, with
        `cementic status` agreeing they were stopped.
        """
        assert is_pid_running(1234) is True

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

    def test_non_object_json(self, temp_dir: Path) -> None:
        """`null` or a list parses fine but would break every caller's .get()."""
        path = temp_dir / "state.json"
        for payload in ("null", "[1, 2]", '"text"'):
            path.write_text(payload)
            assert load_supervisor_state(path) == {}

    def test_unreadable_file(self, temp_dir: Path) -> None:
        """An unreadable state file must not abort start/status/stop."""
        path = temp_dir / "state.json"
        path.write_text("{}")
        path.chmod(0o000)
        try:
            assert load_supervisor_state(path) == {}
        finally:
            path.chmod(0o644)


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


def test_unreadable_start_token_is_unknown_not_a_mismatch():
    """Regression: a token that cannot be *read* -- /proc under hidepid, or a
    process owned by another user -- is not evidence of PID recycling. Treating
    it as one made `cementic stop` skip a live worker, so it never joined the
    not-stopped set and the "nothing left running" branch deleted its state
    files with "cleared stale state" while it kept indexing."""
    with (
        patch("cementic.supervisor.is_pid_running", return_value=True),
        patch("cementic.supervisor.process_start_token", return_value=None),
    ):
        assert is_managed_process_alive(4321, "a-recorded-token") is True


def test_a_genuinely_different_start_token_is_still_a_mismatch():
    """The recycled-PID check must survive the fix above."""
    with (
        patch("cementic.supervisor.is_pid_running", return_value=True),
        patch("cementic.supervisor.process_start_token", return_value="other"),
    ):
        assert is_managed_process_alive(4321, "a-recorded-token") is False


class TestFindPidsByCmdline:
    """Tests for find_pids_by_cmdline: the /proc scan that lets a daemon be
    recovered from the OS when its pid-file record is gone."""

    @staticmethod
    def _write_entry(proc_root: Path, pid: int, argv: list[str]) -> None:
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir(parents=True)
        (pid_dir / "cmdline").write_bytes("\0".join(argv).encode("utf-8") + b"\0")

    def test_matches_by_predicate(self, tmp_path: Path) -> None:
        self._write_entry(tmp_path, 111, ["python", "-m", "llama_cpp.server", "--port", "8083"])
        self._write_entry(tmp_path, 222, ["other-process", "--flag"])
        matches = find_pids_by_cmdline(lambda argv: "llama_cpp.server" in argv, proc_root=tmp_path)
        assert matches == [111]

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        self._write_entry(tmp_path, 111, ["other-process"])
        matches = find_pids_by_cmdline(lambda argv: "llama_cpp.server" in argv, proc_root=tmp_path)
        assert matches == []

    def test_multiple_matches_all_returned(self, tmp_path: Path) -> None:
        self._write_entry(tmp_path, 111, ["a", "match"])
        self._write_entry(tmp_path, 222, ["b", "match"])
        matches = find_pids_by_cmdline(lambda argv: "match" in argv, proc_root=tmp_path)
        assert sorted(matches) == [111, 222]

    def test_non_pid_directories_are_ignored(self, tmp_path: Path) -> None:
        self_dir = tmp_path / "self"
        self_dir.mkdir()
        (self_dir / "cmdline").write_bytes(b"match\0")
        matches = find_pids_by_cmdline(lambda argv: True, proc_root=tmp_path)
        assert matches == []

    def test_owner_uid_mismatch_is_excluded(self, tmp_path: Path) -> None:
        """A process another user owns must never be returned, even if its
        command line matches -- the match criteria alone are not enough."""
        self._write_entry(tmp_path, 111, ["match"])
        with patch("cementic.supervisor.os.getuid", return_value=999999):
            matches = find_pids_by_cmdline(lambda argv: True, proc_root=tmp_path)
        assert matches == []

    def test_missing_proc_root_degrades_to_empty(self, tmp_path: Path) -> None:
        matches = find_pids_by_cmdline(lambda argv: True, proc_root=tmp_path / "does-not-exist")
        assert matches == []

    def test_pid_dir_without_cmdline_is_skipped_not_raised(self, tmp_path: Path) -> None:
        (tmp_path / "111").mkdir()
        matches = find_pids_by_cmdline(lambda argv: True, proc_root=tmp_path)
        assert matches == []

    def test_patching_proc_root_redirects_the_default(self, tmp_path: Path) -> None:
        """Callers that don't pass ``proc_root`` (every real caller) must still
        be redirectable in tests by patching the module constant -- this is
        the mechanism embedding_runtime's daemon-recovery tests rely on to
        fake `/proc` while still calling through the real entry point."""
        self._write_entry(tmp_path, 111, ["match"])
        with patch("cementic.supervisor._PROC_ROOT", tmp_path):
            matches = find_pids_by_cmdline(lambda argv: "match" in argv)
        assert matches == [111]

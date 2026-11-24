"""Contracts on the shipped systemd unit."""

import re
from pathlib import Path

UNIT = Path(__file__).parents[2] / "packaging" / "systemd" / "cementic@.service"


def test_unit_reads_the_configured_daemon_port() -> None:
    text = UNIT.read_text(encoding="utf-8")
    assert "11555" not in text, "the wedge check must not hardcode the default port"

    # The port is read from the real config before the watcher/worker are
    # backgrounded, so a failed read stops the unit instead of arming a probe
    # with an empty port while the daemons run.
    read = text.index("PORT=$$(")
    assert "get_config().llama_cpp.daemon_port" in text[read:]
    assert read < text.index("run watcher")
    assert read < text.index("run worker")
    failure = text[read : text.index("run watcher")]
    assert ">&2" in failure and "exit 1" in failure

    # Every probe URL uses the variable that was read, and none a literal port.
    urls = re.findall(r"http://127\.0\.0\.1:(\S+)", text)
    assert len(urls) == 2
    assert all(url.startswith("$$PORT/") for url in urls), urls

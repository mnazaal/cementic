"""Contracts on the shipped systemd unit."""

from pathlib import Path

UNIT = Path(__file__).parents[2] / "packaging" / "systemd" / "cementic@.service"


def test_unit_reads_the_configured_daemon_port() -> None:
    text = UNIT.read_text(encoding="utf-8")
    assert "11555" not in text, "the wedge check must not hardcode the default port"
    assert "config show" in text and "daemon_port" in text

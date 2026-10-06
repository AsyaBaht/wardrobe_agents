"""CLI: stage 1 failures are reported as a message, never a traceback.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

from typer.testing import CliRunner

from wardrobe_agents import cli
from wardrobe_agents.agents.stylist import StylistError

runner = CliRunner()


def test_an_inverted_temperature_range_is_refused():
    result = runner.invoke(cli.app, ["recommend", "--temp-min", "20", "--temp-max", "5"])

    assert result.exit_code == 1
    assert "--temp-min (20) is above --temp-max (5)" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_stylist_failure_is_a_clean_error(monkeypatch):
    def fail(*args, **kwargs):
        raise StylistError("The stylist returned no usable outfits.")

    monkeypatch.setattr(cli, "run_recommend", fail)
    result = runner.invoke(cli.app, ["recommend", "--temp-min", "5", "--temp-max", "20"])

    assert result.exit_code == 1
    assert "no usable outfits" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)

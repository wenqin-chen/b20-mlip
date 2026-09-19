"""The dft plugin registers exactly its seven commands and every one has help."""

from __future__ import annotations

from typer.testing import CliRunner

from b20mlip.cli import REGISTERED, app
from b20mlip.dft.cli import COMMANDS


def test_registration_and_help() -> None:
    assert (
        REGISTERED["dft"]
        == COMMANDS
        == {"converge", "prep", "run", "collect", "phonons", "e0s", "offsets"}
    )
    runner = CliRunner()
    for name in sorted(COMMANDS):
        res = runner.invoke(app, ["dft", name, "--help"])
        assert res.exit_code == 0 and "[stub]" not in res.output, name
    assert runner.invoke(app, ["dft", "prep"]).exit_code == 2  # --frames/--out required
    assert runner.invoke(app, ["dft", "converge"]).exit_code == 2  # --compound required
    assert (
        runner.invoke(
            app, ["dft", "offsets", "--qe", "a", "--mp", "b", "--out", "c", "--bogus"]
        ).exit_code
        == 2
    )

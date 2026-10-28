# tests/test_cli_llm_choices.py — the CLI's --*-llm flags accept freeform local:<tag> backends.
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_help_documents_local_tag_form():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    assert "local:<tag>" in result.stdout
    assert "--gen-llm" in result.stdout


def test_gen_llm_accepts_prefixed_tag():
    # argparse must not reject a local:<tag> value (no fixed choices= anymore).
    result = subprocess.run(
        [sys.executable, "main.py", "--gen-llm", "local:foo:1b", "--help"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0        # --help short-circuits; the point is no "invalid choice"
    assert "invalid choice" not in result.stderr

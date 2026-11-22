import subprocess
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent / "run.py"


def test_cli_quick_run_prints_both_models(csv_file):
    out = subprocess.run([sys.executable, str(RUN), "--data", str(csv_file), "--quick"],
                         capture_output=True, text=True, check=True).stdout
    assert "Logistic regression classification report" in out
    assert "Random forest classification report" in out
    assert "Test accuracy" in out


def test_cli_can_stop_after_data_preparation(csv_file):
    out = subprocess.run([sys.executable, str(RUN), "--data", str(csv_file), "--skip-models"],
                         capture_output=True, text=True, check=True).stdout
    assert "Classification report" not in out and "classification report" not in out

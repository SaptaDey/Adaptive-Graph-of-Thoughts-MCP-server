import json
import subprocess
from pathlib import Path


def test_vscode_parse_ndjson():
    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "OK" in result.stdout

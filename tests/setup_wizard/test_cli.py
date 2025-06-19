from typer.testing import CliRunner

import agt_setup


def test_cli_creates_env(tmp_path, monkeypatch):
    """
    Test that the CLI creates a `.env` file with correct content and permissions when provided valid input.
    
    Simulates user input for Neo4j connection details, mocks a successful connection, and verifies that the `.env` file is created in the temporary directory with the expected URI and file mode 600.
    """
    env_file = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)

    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://local:7687\nneo4j\npass\nneo4j\n"
    )

    assert result.exit_code == 0
    content = env_file.read_text()
    assert "NEO4J_URI='bolt://local:7687'" in content
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_cli_fails_on_bad_connection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_a, **_k: False)
    runner = CliRunner()
    result = runner.invoke(agt_setup.app, input="x\nx\nx\nx\n")
    assert result.exit_code != 0
    assert not (tmp_path / ".env").exists()

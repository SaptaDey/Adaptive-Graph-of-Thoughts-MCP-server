import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mcp_inspector_executor.py <http|stdio>")
        sys.exit(1)

    mode = sys.argv[1]
    command = []

    if mode == "http":
        command = ["mcp-inspector", "-v", "validate", "http://localhost:8000/mcp"]
    elif mode == "stdio":
        command = [
            "mcp-inspector",
            "-v",
            "validate",
            "stdio",
            "--program",
            "python3",
            "src/adaptive_graph_of_thoughts/main_stdio.py",
        ]
    else:
        print(f"Unknown mode: {mode}. Supported modes are http, stdio.")
        sys.exit(1)

    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=480
        )  # e.g., 480 seconds, or make configurable
        print("MCP Inspector output:")
        print(result.stdout)
        if result.stderr:
            print("MCP Inspector errors:")
            print(result.stderr)
        print(f"MCP Inspector test for {mode} transport passed.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"MCP Inspector command failed with exit code {e.returncode}")
        print("Output:")
        print(e.stdout)
        print("Errors:")
        print(e.stderr)
        print(f"MCP Inspector test for {mode} transport failed.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(
            "Error: mcp-inspector command not found. Make sure it is installed and in your PATH."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

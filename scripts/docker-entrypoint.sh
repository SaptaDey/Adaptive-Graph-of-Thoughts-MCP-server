#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

mode="${1:-http}" # Default to http if no argument provided

echo "Docker entrypoint called with mode: $mode"

if [ "$mode" = "http" ]; then
    echo "Starting HTTP server..."
    exec uvicorn src.asr_got_reimagined.main:app --host 0.0.0.0 --port 8000
elif [ "$mode" = "stdio" ]; then
    echo "Starting STDIO server..."
    exec python -m src.asr_got_reimagined.main_stdio
elif [ "$mode" = "both" ]; then
    # For now, 'both' mode will just run the HTTP server.
    # True concurrent HTTP and STDIO would require a process manager.
    echo "Warning: Mode 'both' currently defaults to running HTTP server only."
    echo "Starting HTTP server..."
    exec uvicorn src.asr_got_reimagined.main:app --host 0.0.0.0 --port 8000
else
    echo "Error: Unknown mode '$mode'. Supported modes are http, stdio, both." >&2
    exit 1
fi

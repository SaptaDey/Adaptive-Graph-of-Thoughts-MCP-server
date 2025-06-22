#!/usr/bin/env python3
"""
Client setup utility for Adaptive Graph of Thoughts MCP Server.
Helps users configure various MCP clients.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent

def load_client_config(client_name: str) -> Dict[str, Any]:
    """Load configuration for a specific client."""
    config_dir = get_project_root() / "config" / "client_configurations"
    config_file = config_dir / f"{client_name}.json"
    
    if not config_file.exists():
        # Try YAML format
        config_file = config_dir / f"{client_name}.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Configuration for {client_name} not found")

def setup_claude_desktop():
    """Setup configuration for Claude Desktop."""
    print("Setting up Claude Desktop MCP configuration...")
    
    try:
        config = load_client_config("claude_desktop")
        
        # Get user inputs
        project_path = input(f"Enter full path to Adaptive-Graph-of-Thoughts-MCP-server directory [{get_project_root()}]: ").strip()
        if not project_path:
            project_path = str(get_project_root())
        
        neo4j_uri = input("Enter Neo4j URI [bolt://localhost:7687]: ").strip()
        if not neo4j_uri:
            neo4j_uri = "bolt://localhost:7687"
        
        neo4j_user = input("Enter Neo4j username [neo4j]: ").strip()
        if not neo4j_user:
            neo4j_user = "neo4j"
        
        neo4j_password = input("Enter Neo4j password: ").strip()
        if not neo4j_password:
            print("Warning: No password provided")
            neo4j_password = "password"
        
        # Update configuration
        stdio_config = config["configurations"][0]["config"]
        stdio_config["mcpServers"]["adaptive-graph-of-thoughts"]["cwd"] = project_path
        stdio_config["mcpServers"]["adaptive-graph-of-thoughts"]["env"]["NEO4J_URI"] = neo4j_uri
        stdio_config["mcpServers"]["adaptive-graph-of-thoughts"]["env"]["NEO4J_USER"] = neo4j_user
        stdio_config["mcpServers"]["adaptive-graph-of-thoughts"]["env"]["NEO4J_PASSWORD"] = neo4j_password
        stdio_config["mcpServers"]["adaptive-graph-of-thoughts"]["env"]["PYTHONPATH"] = f"{project_path}/src"
        
        # Output configuration
        output_file = get_project_root() / "claude_desktop_config.json"
        with open(output_file, 'w') as f:
            json.dump(stdio_config, f, indent=2)
        
        print(f"\nClaude Desktop configuration saved to: {output_file}")
        print("\nTo use this configuration:")
        print("1. Copy the contents of the generated file")
        print("2. Add it to your Claude Desktop MCP settings")
        print("3. Restart Claude Desktop")
        
    except Exception as e:
        print(f"Error setting up Claude Desktop: {e}")
        return False
    
    return True

def setup_vscode():
    """Setup configuration for VS Code."""
    print("Setting up VS Code MCP configuration...")
    
    try:
        config = load_client_config("vscode")
        
        # Get user inputs
        project_path = input(f"Enter full path to Adaptive-Graph-of-Thoughts-MCP-server directory [{get_project_root()}]: ").strip()
        if not project_path:
            project_path = str(get_project_root())
        
        neo4j_uri = input("Enter Neo4j URI [bolt://localhost:7687]: ").strip()
        if not neo4j_uri:
            neo4j_uri = "bolt://localhost:7687"
        
        neo4j_user = input("Enter Neo4j username [neo4j]: ").strip()
        if not neo4j_user:
            neo4j_user = "neo4j"
        
        neo4j_password = input("Enter Neo4j password: ").strip()
        if not neo4j_password:
            print("Warning: No password provided")
            neo4j_password = "password"
        
        # Update configuration
        config["settings"]["mcp.servers"]["adaptive-graph-of-thoughts"]["cwd"] = project_path
        config["settings"]["mcp.servers"]["adaptive-graph-of-thoughts"]["env"]["NEO4J_URI"] = neo4j_uri
        config["settings"]["mcp.servers"]["adaptive-graph-of-thoughts"]["env"]["NEO4J_USER"] = neo4j_user
        config["settings"]["mcp.servers"]["adaptive-graph-of-thoughts"]["env"]["NEO4J_PASSWORD"] = neo4j_password
        config["settings"]["mcp.servers"]["adaptive-graph-of-thoughts"]["env"]["PYTHONPATH"] = f"{project_path}/src"
        
        # Output configuration
        output_file = get_project_root() / "vscode_settings.json"
        with open(output_file, 'w') as f:
            json.dump(config["settings"], f, indent=2)
        
        print(f"\nVS Code configuration saved to: {output_file}")
        print("\nTo use this configuration:")
        print("1. Open VS Code settings (Ctrl/Cmd + ,)")
        print("2. Click 'Open Settings (JSON)' in the top right")
        print("3. Add the contents of the generated file to your settings.json")
        print("4. Install the MCP extension if not already installed")
        print("5. Restart VS Code")
        
    except Exception as e:
        print(f"Error setting up VS Code: {e}")
        return False
    
    return True

def list_available_clients():
    """List all available client configurations."""
    config_dir = get_project_root() / "config" / "client_configurations"
    
    if not config_dir.exists():
        print("No client configurations directory found")
        return
    
    print("Available MCP client configurations:")
    for config_file in config_dir.glob("*.json"):
        client_name = config_file.stem
        print(f"  - {client_name}")
    
    for config_file in config_dir.glob("*.yaml"):
        client_name = config_file.stem
        print(f"  - {client_name}")

def main():
    """Main setup function."""
    if len(sys.argv) < 2:
        print("Adaptive Graph of Thoughts MCP Client Setup")
        print("\nUsage: python client_setup.py <command>")
        print("\nCommands:")
        print("  claude-desktop  - Setup Claude Desktop configuration")
        print("  vscode         - Setup VS Code configuration")
        print("  list           - List available client configurations")
        print("  help           - Show this help message")
        return
    
    command = sys.argv[1].lower()
    
    if command == "claude-desktop":
        setup_claude_desktop()
    elif command == "vscode":
        setup_vscode()
    elif command == "list":
        list_available_clients()
    elif command == "help":
        main()
    else:
        print(f"Unknown command: {command}")
        print("Use 'help' to see available commands")

if __name__ == "__main__":
    main()

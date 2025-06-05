#!/usr/bin/env python3
"""
Helper script to set up the NexusMind MCP connection with Claude Desktop.
This script:
1. Tests if the NexusMind server is running
2. Displays instructions for connecting with Claude Desktop
"""

import json
import logging
import os
import sys

import requests

# Configuration
SERVER_URL = "http://localhost:8000"
MCP_ENDPOINT = f"{SERVER_URL}/mcp"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "claude_mcp_config.json")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_health():
    """Check if the NexusMind server is running by testing the health endpoint."""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ Server is running: Status {health_data['status']}, Version {health_data['version']}")
            return True
        else:
            logger.error(f"❌ Server returned non-200 status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to server: {e}")
        return False

def test_mcp_initialize():
    """
    Sends an initialize request to the MCP endpoint to verify its availability.
    
    Returns:
        True if the MCP endpoint responds successfully with a result; False otherwise.
    """
    init_payload = {
        "jsonrpc": "2.0",
        "id": "setup-script-1",
        "method": "initialize",
        "params": {
            "client_info": {
                "client_name": "NexusMind Setup Script"
            },
            "process_id": 12345
        }
    }

    try:
        response = requests.post(
            MCP_ENDPOINT,
            json=init_payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("result"):
                logger.info("✅ MCP endpoint initialized successfully")
                logger.info("   Server name: NexusMind MCP Server")
                logger.info("   Server version: 0.1.0")
                logger.info("   MCP version: 2024-11-05")
                return True
            else:
                logger.error(f"❌ MCP endpoint returned an error: {json.dumps(result.get('error', {}), indent=2)}")
                return False
        else:
            logger.error(f"❌ MCP endpoint returned status code: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing MCP endpoint: {e}")
        return False

def check_config_file():
    """
    Validates the existence and structure of the MCP configuration file.
    
    Checks if the configuration file exists, is valid JSON, and contains the required
    'connection' and 'endpoint' fields with the expected endpoint value.
    
    Returns:
        True if the configuration file is present and valid; False otherwise.
    """
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"❌ MCP configuration file not found: {CONFIG_FILE}")
        return False

    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)

        # Validate essential fields
        if (
            "connection" in config and
            "endpoint" in config["connection"] and
            config["connection"]["endpoint"] == MCP_ENDPOINT
        ):
            logger.info("✅ MCP configuration file is valid")
            return True
        else:
            logger.warning("⚠️  MCP configuration file has issues:")
            if "connection" not in config:
                logger.warning("   Missing 'connection' field")
            elif "endpoint" not in config["connection"]:
                logger.warning("   Missing 'endpoint' field in connection")
            elif config["connection"]["endpoint"] != MCP_ENDPOINT:
                logger.warning(f"   Endpoint mismatch: {config['connection']['endpoint']} vs {MCP_ENDPOINT}")
            return False
    except json.JSONDecodeError:
        logger.error("❌ MCP configuration file is not valid JSON")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking MCP configuration: {e}")
        return False

def display_instructions():
    """Display instructions for connecting with Claude Desktop."""
    logger.info("\n" + "=" * 60)
    logger.info("CLAUDE DESKTOP CONNECTION INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("\n1. Open Claude Desktop")
    logger.info("2. Go to Settings (gear icon)")
    logger.info("3. Navigate to Tools or Integrations")
    logger.info("4. Add a new Tool/Integration")
    logger.info("5. Import the MCP configuration file from:")
    logger.info(f"   {os.path.abspath(CONFIG_FILE)}")
    logger.info("\nAlternative manual setup:")
    logger.info("- Name: NexusMind MCP Integration")
    logger.info("- Description: Scientific reasoning with Graph of Thoughts")
    logger.info(f"- Endpoint: {MCP_ENDPOINT}")
    logger.info("- Method: POST")
    logger.info("- Headers: Content-Type: application/json")
    logger.info("\nFor complete instructions, see:")
    logger.info("docs/claude_desktop_integration.md")
    logger.info("\n" + "=" * 60)

def main():
    """
    Runs the NexusMind MCP setup process, performing server checks, configuration validation, and displaying connection instructions.
    
    Executes a sequence of steps to verify server health, test the MCP endpoint, validate the MCP configuration file, and provide instructions for connecting Claude Desktop to the NexusMind server. Exits the program if the server is not running.
    """
    logger.info("\n=== NexusMind MCP Setup ===\n")

    # Step 1: Check if server is running
    logger.info("Step 1: Checking if NexusMind server is running...")
    if not check_health():
        logger.warning("\n⚠️  WARNING: Server not running. Please start the server and try again.")
        logger.warning("   Docker command: docker-compose up -d")
        sys.exit(1)

    # Step 2: Test MCP endpoint
    logger.info("\nStep 2: Testing MCP endpoint...")
    if not test_mcp_initialize():
        logger.warning("\n⚠️  WARNING: MCP endpoint not working properly.")
        logger.warning("   Please check the server logs for errors.")

    # Step 3: Check config file
    logger.info("\nStep 3: Checking MCP configuration file...")
    check_config_file()

    # Step 4: Display instructions
    logger.info("\nStep 4: Connection instructions...")
    display_instructions()

    logger.info("\nSetup complete! You can now connect Claude Desktop to the NexusMind server.")
    logger.info("Test the integration by asking a scientific reasoning question.")

if __name__ == "__main__":
    main()

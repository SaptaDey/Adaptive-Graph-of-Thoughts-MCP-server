## Accessing Neo4j in Development (Windows with WSL2)

When running the application stack using Docker Desktop on Windows with the WSL2 backend, the Neo4j database service should be directly accessible from your Windows host environment. Modern versions of Docker Desktop handle the necessary port forwarding from `localhost` on Windows to the services running within the WSL2 containers.

**Neo4j Service Access:**

*   **Neo4j Browser:** You can access the Neo4j web browser interface by navigating to:
    `http://localhost:7474`
    This interface allows you to execute Cypher queries, explore the graph visually, and manage the database.

*   **Bolt Protocol URI:** For connecting to Neo4j from client applications (e.g., Python scripts using the `neo4j` driver, or other database tools), use the following Bolt URI:
    `neo4j://localhost:7687`

**Default Credentials:**

The Neo4j service is configured with default credentials in the `docker-compose.yml` file. Unless you have modified these, they are:
*   **Username:** `neo4j`
*   **Password:** `password`

You will be prompted for these when connecting via the Neo4j Browser or a Bolt client.

**Troubleshooting Connectivity:**

In most cases, accessing Neo4j via `localhost` should work seamlessly. However, if you encounter issues connecting to Neo4j on `localhost:7474` or `localhost:7687`:

1.  **Docker Desktop Version & Settings:** Ensure you are running a recent version of Docker Desktop. Older versions might have limitations with WSL2 networking. Check your Docker Desktop settings, particularly those related to WSL2 integration and networking, to ensure they are configured correctly.
2.  **Firewall Configuration:** Your Windows firewall or any third-party firewall software could potentially block the connection. Temporarily disabling the firewall (for testing purposes only) or adding explicit rules to allow traffic on ports `7474` and `7687` might help identify if this is the issue.
3.  **Other Network Configurations:** Complex network setups, VPNs, or proxy servers could interfere with localhost forwarding.
4.  **WSL2 Port Forwarding (Fallback):** In rare cases, especially with older Docker Desktop versions or specific WSL2 network configurations, `localhost` forwarding might not work as expected. If you've ruled out other causes, you might need to manually set up port forwarding from your Windows host to the WSL2 IP address for the container. This can be done using the `netsh interface portproxy` command in Windows. For detailed instructions on this, you can search online for "WSL2 port forwarding netsh" or similar terms. This is generally considered a last resort.

If you continue to experience issues, consult the Docker Desktop documentation for WSL2 networking troubleshooting.

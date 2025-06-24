from __future__ import annotations

"""Simple secret management helper supporting multiple providers."""

import os
from typing import Optional

from loguru import logger


class SecretManager:
    """Retrieve secrets from various backends."""

    def __init__(self, provider: str = "env") -> None:
        self.provider = provider.lower()

    def get_secret(self, name: str) -> Optional[str]:
        """Return the secret value for ``name`` or ``None`` if not found."""
        if self.provider == "env":
            return os.getenv(name)
        if self.provider == "aws":  # pragma: no cover - requires boto3
            try:
                import boto3  # type: ignore

                client = boto3.client("secretsmanager")
                resp = client.get_secret_value(SecretId=name)
                return resp.get("SecretString")
            except Exception as exc:  # pragma: no cover - best effort
                logger.error(f"Failed to load AWS secret {name}: {exc}")
                return None
        if self.provider == "gcp":  # pragma: no cover - requires google libs
            try:
                from google.cloud import secretmanager  # type: ignore

                client = secretmanager.SecretManagerServiceClient()
                project_id = os.getenv("GCP_PROJECT_ID")
                if not project_id:
                    raise RuntimeError("GCP_PROJECT_ID not set")
                secret_path = f"projects/{project_id}/secrets/{name}/versions/latest"
                response = client.access_secret_version(name=secret_path)
                return response.payload.data.decode("UTF-8")
            except Exception as exc:  # pragma: no cover
                logger.error(f"Failed to load GCP secret {name}: {exc}")
                return None
        if self.provider == "vault":  # pragma: no cover - requires hvac
            try:
                import hvac  # type: ignore

                client = hvac.Client(
                    url=os.getenv("VAULT_ADDR"), token=os.getenv("VAULT_TOKEN")
                )
                read_resp = client.secrets.kv.v2.read_secret_version(path=name)
                return read_resp["data"]["data"].get("value")
            except Exception as exc:  # pragma: no cover
                logger.error(f"Failed to load Vault secret {name}: {exc}")
                return None
        logger.warning(f"Unknown secrets provider: {self.provider}")
        return None


def load_external_secrets() -> None:
    """Load known secrets into environment variables if missing."""

    provider = os.getenv("SECRETS_PROVIDER")
    if not provider or provider.lower() == "env":
        return

    manager = SecretManager(provider)
    for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEO4J_PASSWORD"]:
        if os.getenv(var):
            continue
        secret_name = os.getenv(f"{var}_SECRET_NAME", var)
        secret = manager.get_secret(secret_name)
        if secret:
            os.environ[var] = secret
            logger.debug(f"Loaded secret for {var} from {provider}")

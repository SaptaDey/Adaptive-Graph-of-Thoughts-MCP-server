import httpx
from typing import Optional, Dict, Any
from loguru import logger
from adaptive_graph_of_thoughts.config import Settings # May not be used initially but good for future config

class BaseAPIClientError(Exception):
    """Base exception for API client errors."""
    pass

class APIRequestError(BaseAPIClientError):
    """Indicates an error during the API request (network, timeout, etc.)."""
    pass

class APIHTTPError(BaseAPIClientError):
    """Indicates an HTTP error response from the API (4xx, 5xx)."""
    def __init__(self, status_code: int, response_content: Any, message: Optional[str] = None):
        """
        Initializes an APIHTTPError with the HTTP status code, response content, and an optional message.
        
        Args:
            status_code: The HTTP status code returned by the API.
            response_content: The content of the HTTP response.
            message: An optional custom error message. If not provided, a default message is generated.
        """
        self.status_code = status_code
        self.response_content = response_content
        self.message = message or f"API returned HTTP {status_code}"
        super().__init__(self.message)

class AsyncHTTPClient:
    """
    A base asynchronous HTTP client using httpx.
    Manages an httpx.AsyncClient instance for making requests.
    """
    def __init__(
        self,
        base_url: str,
        settings: Optional[Settings] = None, # For future use (e.g., global timeouts)
        api_key: Optional[str] = None, # Specific clients might handle API keys differently
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the asynchronous HTTP client with a base URL, optional API key, and default headers.
        
        Args:
            base_url: The base URL for all HTTP requests made by this client.
            default_headers: Additional headers to include with every request.
        
        The client is configured with a default user-agent, JSON accept header, and a 20-second timeout (10 seconds for connection). An API key and settings can be stored for use by subclasses or future extensions.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key # Store if needed for common auth, though often header-specific
        self.settings = settings

        # Prepare headers
        headers = {
            "User-Agent": "AdaptiveGraphOfThoughts/1.0",
            "Accept": "application/json",
        }
        if default_headers:
            headers.update(default_headers)

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(20.0, connect=10.0), # Increased timeout
            follow_redirects=True,
        )
        logger.debug(f"AsyncHTTPClient initialized for base URL: {self.base_url}")

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None
    ) -> httpx.Response:
        """
        Performs an asynchronous HTTP GET request to the specified API endpoint.
        
        Args:
        	endpoint: The relative path of the API endpoint to request.
        	params: Optional query parameters to include in the request.
        	headers: Optional headers to include in the request.
        
        Returns:
        	The HTTP response object if the request is successful.
        
        Raises:
        	APIHTTPError: If the API responds with an HTTP error status (4xx or 5xx).
        	APIRequestError: If a network or request-related error occurs.
        """
        endpoint = endpoint.lstrip('/')
        logger.debug(f"GET request to {self.base_url}/{endpoint} with params: {params}")
        try:
            response = await self.client.get(url=f"/{endpoint}", params=params, headers=headers)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            logger.debug(f"GET request to {self.base_url}/{endpoint} successful with status: {response.status_code}")
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {e.request.url}: {e.response.text[:200]}")
            raise APIHTTPError(status_code=e.response.status_code, response_content=e.response.text, message=str(e)) from e
        except httpx.RequestError as e:
            logger.error(f"Request error for {e.request.url}: {str(e)}")
            raise APIRequestError(f"Request error for {e.request.url}: {str(e)}") from e

    async def post(
        self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None
    ) -> httpx.Response:
        """
        Sends an asynchronous HTTP POST request to the specified API endpoint.
        
        Args:
        	endpoint: The relative path of the API endpoint to send the request to.
        	json_data: Optional dictionary to send as JSON in the request body.
        	data: Optional dictionary to send as form data in the request body.
        	headers: Optional dictionary of additional headers to include in the request.
        
        Returns:
        	The HTTP response object if the request is successful.
        
        Raises:
        	APIHTTPError: If the API responds with an HTTP error status (4xx or 5xx).
        	APIRequestError: If a network or request-related error occurs.
        """
        endpoint = endpoint.lstrip('/')
        logger.debug(f"POST request to {self.base_url}/{endpoint}")
        try:
            response = await self.client.post(url=f"/{endpoint}", json=json_data, data=data, headers=headers)
            response.raise_for_status()
            logger.debug(f"POST request to {self.base_url}/{endpoint} successful with status: {response.status_code}")
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {e.request.url}: {e.response.text[:200]}")
            raise APIHTTPError(status_code=e.response.status_code, response_content=e.response.text, message=str(e)) from e
        except httpx.RequestError as e:
            logger.error(f"Request error for {e.request.url}: {str(e)}")
            raise APIRequestError(f"Request error for {e.request.url}: {str(e)}") from e

    async def close(self):
        """
        Closes the underlying asynchronous HTTP client.
        
        This method should be called to release network resources when the client is no longer needed.
        """
        logger.debug(f"Closing AsyncHTTPClient for base URL: {self.base_url}")
        await self.client.aclose()

    async def __aenter__(self):
        """
        Enters the asynchronous context manager and returns the client instance.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures the HTTP client is properly closed when exiting an asynchronous context.
        """
        await self.close()

if __name__ == '__main__':
    # Example Usage (for testing the base client directly if needed)
    import asyncio

    async def main():
        # This example won't run without a live server, it's for structure.
        # Replace with a public API for actual testing if desired.
        # Example: client = AsyncHTTPClient(base_url="https://jsonplaceholder.typicode.com")
        # try:
        #     async with client:
        #         response = await client.get(endpoint="/todos/1")
        #         data = response.json()
        #         logger.info(f"Example GET response data: {data}")
        # except Exception as e:
        #     logger.error(f"Example usage error: {e}")
        pass # Placeholder for direct execution example

    if __name__ == '__main__': # This nested check is a bit unusual, ensure it's what you intend
        # asyncio.run(main()) # Commented out to prevent execution in subtask
        pass

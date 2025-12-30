"""
api_client.py

Simple API client that accepts an API key directly (no .env).
Uses 'requests'. Install: pip install requests
"""
import os
from typing import Any, Dict, Optional

import requests


class XAIClient:
    """
    Minimal API client wrapper.
    - base_url: API base URL (e.g. 'https://api.example.com/v1')
    - api_key: API key string (required)
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        if not api_key:
            raise ValueError("api_key is required for XAIClient (no .env used).")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "XAIClient/1.0",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=self._headers(),
            params=params,
            json=json,
            timeout=self.timeout,
        )
        # raise HTTPError on bad status
        resp.raise_for_status()

        # Try to return parsed JSON, otherwise raw text
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def ping(self) -> Any:
        """Call GET /ping (useful as a health check if your API supports it)."""
        return self.request("GET", "/ping")

    def chat(self, user_text: str, conversation_id: Optional[str] = None) -> Any:
        """
        Example convenience method to send chat text.
        Adjust endpoint/payload to match your API.
        """
        payload = {"message": user_text}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        return self.request("POST", "/chat", json=payload)


if __name__ == "__main__":
    # quick local test (not used when imported)
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base URL for API (e.g. https://api.example.com)")
    parser.add_argument("--key", required=True, help="API key (pass directly; no .env)")
    args = parser.parse_args()

    client = XAIClient(base_url=args.base, api_key=args.key)
    try:
        print("Pinging API...")
        print(client.ping())
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        raise

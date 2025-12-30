"""
chat_manager.py

Simple adapter between your application and XAIClient.
It exposes get_response(text) -> str which calls the API's chat endpoint.
"""

from typing import Any, Optional

from api_client import XAIClient


class ChatManager:
    def __init__(self, client: XAIClient, conversation_id: Optional[str] = None) -> None:
        self.client = client
        self.conversation_id = conversation_id

    def get_response(self, user_text: str) -> str:
        """
        Send the user text to the API (via XAIClient.chat) and return the reply text.
        Adjust parsing depending on the API response shape.
        """
        try:
            resp: Any = self.client.chat(user_text, conversation_id=self.conversation_id)
        except Exception as exc:
            # graceful fallback
            return f"Sorry, I couldn't reach the API: {exc}"

        # Expecting dict-like response. Try common keys.
        if isinstance(resp, dict):
            # common patterns: {"reply": "..."} or {"message": {"text":"..."}}
            if "reply" in resp and isinstance(resp["reply"], str):
                return resp["reply"]
            if "message" in resp and isinstance(resp["message"], dict):
                text = resp["message"].get("text") or resp["message"].get("reply")
                if isinstance(text, str):
                    return text
            # maybe the API returns {'choices':[{'text':'...'}]}
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
            # generic fallback: stringify the response
            return str(resp)

        # If response is plain text, return it
        if isinstance(resp, str):
            return resp

        return "Received an unexpected response format from the API."

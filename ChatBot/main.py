"""
main.py

Entry point. Uses Option B: API key placed directly in code (no .env required).
Edit API_KEY and BASE_URL below with your real credentials.
"""

from api_client import XAIClient
from chat_manager import ChatManager
from stt import SpeechToText
from tts import TextToSpeech


# -----------------------------
#  OPTION B: Set your API key here (no .env)
# -----------------------------
API_KEY = "API KEY"            # <<< REPLACE with your real API key
BASE_URL = "https://api.example.com"     # <<< REPLACE with your real API base URL
# -----------------------------

def main() -> None:
    # Create API client using direct key (no .env)
    client = XAIClient(base_url=BASE_URL, api_key=API_KEY)

    # Initialize modules
    stt = SpeechToText()
    tts = TextToSpeech()
    chat_manager = ChatManager(client)

    print("Voice Assistant Running...")
    print("Say something (Ctrl+C to exit)")

    try:
        while True:
            user_text = stt.listen()
            if not user_text:
                # nothing captured; loop again
                continue

            print(f"You said: {user_text}")

            bot_response = chat_manager.get_response(user_text)
            print(f"Bot: {bot_response}")

            tts.speak(bot_response)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    main()

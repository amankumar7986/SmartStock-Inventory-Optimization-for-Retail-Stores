"""
tts.py

Text-to-Speech helper using pyttsx3 (offline).
Install: pip install pyttsx3

Note: On macOS, pyttsx3 uses the system voices and typically works out of the box.
"""

import sys
from typing import Optional

try:
    import pyttsx3
except Exception as e:
    raise ImportError("pyttsx3 is required. Install with: pip install pyttsx3") from e


class TextToSpeech:
    def __init__(self, rate: Optional[int] = None, volume: Optional[float] = None) -> None:
        self.engine = pyttsx3.init()
        if rate is not None:
            self.engine.setProperty("rate", rate)
        if volume is not None:
            # volume: float 0.0 - 1.0
            self.engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}", file=sys.stderr)

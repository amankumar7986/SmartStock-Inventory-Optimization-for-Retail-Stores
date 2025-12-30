"""
stt.py

Speech-to-Text helper using the SpeechRecognition library.
Install: pip install SpeechRecognition
For microphone support you'll typically need PyAudio (pip install pyaudio)
On macOS use: brew install portaudio; then pip install pyaudio
"""

import sys
from typing import Optional

try:
    import speech_recognition as sr
except Exception as e:
    raise ImportError(
        "speech_recognition is required. Install with: pip install SpeechRecognition"
    ) from e


class SpeechToText:
    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.8) -> None:
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold

    def listen(self) -> Optional[str]:
        """
        Listen from the default microphone and return recognized text.
        Returns None on no speech / unrecognized / error.
        """
        try:
            with sr.Microphone() as mic:
                print("Listening (speak now)...")
                audio = self.recognizer.listen(mic, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            print("Listening timed out waiting for phrase to start.")
            return None
        except Exception as e:
            print(f"Microphone error: {e}", file=sys.stderr)
            return None

        # Recognize using Google's free API (online). You can replace with other recognizers.
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected STT error: {e}")
            return None

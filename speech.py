from gtts import gTTS
import pygame
import tempfile
import threading
import os

def speak(text):
    """Non-blocking speak — detection loop continues while audio plays."""
    thread = threading.Thread(target=_speak_worker, args=(text,))
    thread.daemon = True
    thread.start()

def _speak_worker(text):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        tts = gTTS(text=text, lang='en')
        file_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        tts.save(file_path)

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()
        os.remove(file_path)

    except Exception as e:
        print(f"[Speech Error] {e}")
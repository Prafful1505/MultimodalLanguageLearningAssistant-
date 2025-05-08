import os
import pygame
from gtts import gTTS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize pygame mixer
pygame.mixer.init()

def text_to_speech_with_gtts(input_text, output_filepath):
    """Convert text to speech using gTTS and save/play the audio."""
    try:
        if not input_text:
            raise ValueError("No input text provided for TTS")
        audioobj = gTTS(text=input_text, lang="en", slow=False)
        audioobj.save(output_filepath)
        logging.info(f"Audio saved to {output_filepath}")
        
        # Play the audio
        play_audio_with_pygame(output_filepath)
    except Exception as e:
        logging.error(f"Error in gTTS: {e}")
        raise

def play_audio_with_pygame(output_filepath):
    """Play an MP3 file using pygame."""
    try:
        if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) == 0:
            raise FileNotFoundError(f"Audio file {output_filepath} doesn't exist or is empty")
        pygame.mixer.music.load(output_filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        logging.error(f"Error playing audio: {e}")
        raise
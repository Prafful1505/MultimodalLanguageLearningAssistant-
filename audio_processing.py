import os
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_microphones():
    """List available microphones and their indices."""
    try:
        mics = sr.Microphone.list_microphone_names()
        logging.info("Available microphones:")
        for index, name in enumerate(mics):
            logging.info(f"Microphone {index}: {name}")
        return mics
    except Exception as e:
        logging.error(f"Error listing microphones: {e}")
        return []

def record_audio(file_path, timeout=15, phrase_time_limit=10, retries=3):
    """Record audio from the microphone and save as MP3 with 16-bit quality."""
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 100  # Lowered for sensitivity
        recognizer.dynamic_energy_threshold = True

        # List available microphones
        mics = list_microphones()
        if not mics:
            raise Exception("No microphones detected. Please connect a microphone.")

        # Use microphone index 5 (sof-hda-dsp: - (hw:0,7))
        mic_index = 5
        logging.info(f"Using microphone: {mics[mic_index]} (index {mic_index})")

        for attempt in range(retries):
            try:
                with sr.Microphone(sample_rate=16000, device_index=mic_index) as source:
                    logging.info(f"Attempt {attempt + 1}/{retries}: Adjusting for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=2)
                    logging.info("Start speaking now (you have 15 seconds to begin)...")

                    # Record audio
                    audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    logging.info("Recording complete.")

                    # Save as WAV for testing, then convert to MP3
                    wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)  # 16-bit
                    audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                    audio_segment.export(file_path, format="mp3", bitrate="128k")
                    logging.info(f"Audio saved to {file_path}")

                    # Verify audio file
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        return
                    else:
                        raise Exception("Audio file is empty or not created.")
            except sr.WaitTimeoutError:
                logging.warning(f"Attempt {attempt + 1} timed out. Retrying...")
                if attempt == retries - 1:
                    raise Exception("No speech detected after multiple attempts. Speak louder or check microphone.")
            except sr.UnknownValueError:
                raise Exception("No speech detected. Speak clearly and try again.")
            except sr.RequestError as e:
                raise Exception(f"Microphone error: {e}. Check microphone connection and permissions.")
    except Exception as e:
        logging.error(f"Error recording audio: {e}")
        raise

def transcribe_with_groq(audio_filepath, groq_api_key):
    """Transcribe audio using Groq's Whisper model."""
    try:
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file {audio_filepath} not found")
        client = Groq(api_key=groq_api_key)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise
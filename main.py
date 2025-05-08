import os
import time
import gradio as gr
from dotenv import load_dotenv
import logging
from audio_processing import record_audio, transcribe_with_groq
from tts import text_to_speech_with_gtts
from language_feedback import get_language_feedback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Ensure output directory
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)

def process_audio(audio_filepath):
    """Process audio input to generate transcription and language feedback."""
    try:
        # Handle invalid or missing audio
        if not audio_filepath or not isinstance(audio_filepath, str) or not os.path.exists(audio_filepath):
            logging.warning("Invalid or no audio filepath. Recording new audio.")
            audio_filepath = os.path.join(output_dir, f"user_audio_{int(time.time())}.mp3")
            record_audio(file_path=audio_filepath, timeout=15, phrase_time_limit=10, retries=3)

        # Transcribe audio
        logging.info("Transcribing audio...")
        transcription = transcribe_with_groq(audio_filepath=audio_filepath, groq_api_key=GROQ_API_KEY)

        # Get language feedback
        logging.info("Generating language feedback...")
        feedback = get_language_feedback(transcription=transcription, groq_api_key=GROQ_API_KEY)

        # Generate TTS for feedback
        tts_filepath = os.path.join(output_dir, f"feedback_{int(time.time())}.mp3")
        text_to_speech_with_gtts(input_text=feedback, output_filepath=tts_filepath)

        return transcription, feedback, tts_filepath
    except Exception as e:
        logging.error(f"Error in process_audio: {e}")
        return f"Error: {e}", "", ""

# Create Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record your sentence")
    ],
    outputs=[
        gr.Textbox(label="Your Transcription"),
        gr.Textbox(label="Feedback and Practice Prompt"),
        gr.Audio(label="Spoken Feedback")
    ],
    title="Multimodal Language Learning Assistant",
    description="Record a sentence in English to receive feedback on grammar and pronunciation, along with a practice prompt."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(debug=True, server_port=7861)
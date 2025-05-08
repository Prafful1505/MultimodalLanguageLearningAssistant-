import os
from groq import Groq
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_language_feedback(transcription, groq_api_key):
    """Analyze transcription for grammar and pronunciation feedback using Groq's LLaMA model."""
    try:
        client = Groq(api_key=groq_api_key)
        system_prompt = (
            "You are a language learning assistant. Analyze the user's spoken sentence for grammar and pronunciation. "
            "Provide clear, concise feedback on grammar errors, word choice, and potential pronunciation issues based on the transcription. "
            "Suggest corrections and provide a follow-up practice prompt to reinforce learning. "
            "Format the response as:\n"
            "Feedback: [Your feedback on grammar and pronunciation]\n"
            "Correction: [Corrected sentence or suggestion]\n"
            "Practice Prompt: [A new sentence or question for the user to practice]"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this sentence: {transcription}"}
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in language feedback: {e}")
        raise
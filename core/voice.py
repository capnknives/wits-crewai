# core/voice.py
# Captures voice input and uses Whisper to transcribe

import sounddevice as sd
import numpy as np
import whisper

_model = None # Global model cache

def get_model(model_name="base"):
    """Loads and caches the Whisper model."""
    global _model
    if _model is None:
        try:
            print(f"[VOICE] Loading Whisper model: {model_name}...")
            _model = whisper.load_model(model_name)
            print(f"[VOICE] Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"[VOICE ERROR] Failed to load Whisper model '{model_name}': {e}")
            # Optionally, try loading a fallback model like 'tiny' if the specified one fails
            # Or raise the exception to be handled by the caller
            raise # Re-raise the exception to notify the caller
    return _model

def get_voice_input(duration=5, model_name="base", samplerate=16000, channels=1, dtype='float32', whisper_fp16=False):
    """
    Captures audio from the microphone and transcribes it using Whisper.
    Args:
        duration (int): Recording duration in seconds.
        model_name (str): Name of the Whisper model to use.
        samplerate (int): Audio sample rate.
        channels (int): Number of audio channels.
        dtype (str): Audio data type.
        whisper_fp16 (bool): Whether to use FP16 for Whisper transcription (if supported).
    Returns:
        str: Transcribed text, or an empty string if an error occurs or no speech is detected.
    """
    try:
        print(f"[VOICE] Recording for {duration} seconds (model: {model_name}, fp16: {whisper_fp16})...")
        # Ensure sounddevice defaults are set if not passed (though main.py sets them)
        sd.default.samplerate = samplerate
        sd.default.channels = channels
        sd.default.dtype = dtype

        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype=dtype)
        sd.wait()  # Wait until recording is finished
        audio_data = np.squeeze(recording) # Remove single-dimensional entries from the shape of an array.

        if np.max(np.abs(audio_data)) < 0.01: # Basic silence check
            print("[VOICE] Low audio input detected, possibly silence.")
            # return "" # Uncomment if you want to return early on silence

        model = get_model(model_name)
        if model is None: # If model loading failed in get_model
            print("[VOICE ERROR] Whisper model not available for transcription.")
            return ""

        print("[VOICE] Transcribing...")
        # Note: The 'fp16' option in OpenAI's Whisper CLI is for the model weights type during conversion,
        # not directly a transcription-time option in the same way for all backends.
        # The library's transcribe function handles precision internally.
        # For CPU, fp16 is often ignored or defaults to False with a warning.
        # If using a GPU and a version of Whisper/PyTorch that supports it, it might be used.
        # The common way to set fp16 is in the model.transcribe() call if the library supports it.
        # openai-whisper's `transcribe` method takes `fp16` as an argument.
        result = model.transcribe(audio_data, fp16=whisper_fp16)
        transcribed_text = result.get("text", "").strip()

        if not transcribed_text:
            print("[VOICE] Transcription result is empty.")
        return transcribed_text

    except Exception as e:
        print(f"[VOICE ERROR] during get_voice_input: {e}")
        return ""
import pyaudio
import numpy as np
from transformers import pipeline

# Charger le modèle ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en"  # Modèle exclusivement anglais
)

# Paramètres audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Initialiser PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Parlez... (Appuyez sur Ctrl+C pour arrêter)")

try:
    frames = []

    # Enregistrement de l'audio
    while True:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

except KeyboardInterrupt:
    print("\nEnregistrement arrêté.")
    audio_data = np.concatenate(frames)

    # Transcription
    print("Transcription en cours...")
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    transcription = asr_pipeline(audio_float32)["text"]
    print(f"Texte transcrit : {transcription}")

finally:
    # Libération des ressources
    stream.stop_stream()
    stream.close()
    audio.terminate()
import pyaudio
import numpy as np
import torch
from transformers import pipeline

# Charger le modèle ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",  # Modèle exclusivement anglais
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Paramètres audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
DURATION = 5

def audio_listening():
    # Initialiser PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Parlez... ")

    #try:
    frames = []

    # Enregistrement de l'audio
#        while True:
#             data = stream.read(CHUNK)
#            frames.append(np.frombuffer(data, dtype=np.int16))

    # Record in chunks and append data to frames
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))  # Convert to NumPy array


#except KeyboardInterrupt:
    print("\nEnregistrement arrêté.")
    audio_data = np.concatenate(frames)

    # Transcription
    print("Transcription en cours...")
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    transcription = asr_pipeline(audio_float32)["text"]
    #print(f"Texte transcrit : {transcription}")

#finally:
    # Libération des ressources
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return transcription

t = audio_listening()
print(f"You just said: {t}")
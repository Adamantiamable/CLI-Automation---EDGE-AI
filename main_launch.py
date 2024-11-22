import automatic_audio_transcription as audio

#First step : audio transcription
texte = audio.audio_listening()

print(f"Texte : {texte}")
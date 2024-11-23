import os
import subprocess
import dotenv
from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer
import concurrent.futures
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import automatic_audio_transcription as audio
from paths_command_extraction import generate_file_descriptions, find_nearest_command

def main():
    print("Start")

    input_text = audio.audio_listening()

    descriptions, embeddings = generate_file_descriptions()

    print(input_text)

    # query = "I want to look at the contents of my inference.py file"
    query = input_text
    #query = "I want to move my extraction file python file to the destination output folder"
    result = find_nearest_command(query, descriptions, embeddings)

    print(f"Query: {query}")
    print(f"Command: {result['command']}")
    print(f"Description: {result['description']}")
    print(f"Arguments: {result['arguments']}")
    print(f"Generated command: {result['generated_command']}")
    print(f"Confidence: {result['confidence']:.3f}")

    if result['command'] == 'type':
        result['command'] = 'cat'
    actual_command = result['command'] + ' ' + result['arguments']['file']
    
    subprocess_output = subprocess.run(
        actual_command.split(' '),
        capture_output=True
    )

    if subprocess_output.returncode == 0:
        print('SUCCESS')
        print(subprocess_output.stdout)
        print("Command executed successfuly")
    else:
        print('FAIL')
        print('stdout:')
        print(subprocess_output.stdout)
        print('stderr:')
        print(subprocess_output.stderr)
    

if __name__ == "__main__":
    main()
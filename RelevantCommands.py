import os
import subprocess
import dotenv
from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer
import concurrent.futures

from torch import cosine_similarity

dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# Text Provided

text_to_use_as_input = "I want to find a directory in my system"

# Get AI text output
def get_AI_output(prompt, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try : 
            client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key = API_KEY,
            )
            
            completion = client.chat.completions.create(
            extra_headers={},
            model="google/gemini-flash-1.5-8b",
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ]
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            
            # If it's the last attempt, re-raise the error
            if attempt == max_retries - 1:
                raise

            # Wait before retrying
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

## Build list of relevant commands based on commands available on system
def commands_list_generator():
    try:
        result = subprocess.run(
            ["bash", "-c", "compgen -c"],  # Run compgen -c in a bash shell
            text=True,                    # Ensure output is in string format
            capture_output=True,          # Capture stdout and stderr
            check=True                    # Raise exception on error
        )
        commands = result.stdout.splitlines()  # Split the output into a list of commands
        print(f"Retrieved {len(commands)} commands")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

    # Filter the lines :

    commands_to_remove = {
        "if", "then", "else", "elif", "fi", "case", "esac", "for", "select", "while", "until", 
        "do", "done", "in", "function", "time", "{", "}", "!", "[[", "]]", "coproc", ".", ":", "["
    }

    filtered_commands = [cmd for cmd in commands if cmd not in commands_to_remove]
    extensions_to_remove = (".dll", ".exe", ".py", ".DLL", ".sys", ".cpl", ".png",
    ".dat", ".mof", ".exe", ".config", ".ps1xml",
    ".NLS", ".xml", ".log", ".txt", ".ax", ".com", ".xlm", ".gif")

    # Save the list of commands : within a file :

    with open('filtered_commands.txt', 'w') as file:
        for command in commands:
            if (command not in commands_to_remove) & (not command.endswith(extensions_to_remove)):
                file.write(command + '\n')
    
    #Additional refinement mechanisms


if not os.path.exists("filtered_commands.txt"):
    commands_list_generator(); 
else :
    print("Command list generated & ready to use")

## Define for each command 1. A description 2. Associated arguments 

with open('filtered_commands.txt') as file:
    commands = file.read().splitlines()

prompts = []
commands_with_descriptions = []
def process_command(command):
    prompt = f"`{command}`\n provide a 3 sentences description of the above terminal command and list its usecases"
    description = get_AI_output(prompt)
    print(f"Prompt: {prompt!r}, Generated text: {description!r}")
    return (command, description)

def track_progress(futures, total):
    completed = 0
    start_time = time.time()
    for future in concurrent.futures.as_completed(futures):
        completed += 1
        elapsed_time = time.time() - start_time
        remaining = total - completed
        estimated_total_time = (elapsed_time / completed) * total
        estimated_remaining_time = estimated_total_time - elapsed_time
        print(f"Completed {completed}/{total}. Estimated remaining time: {estimated_remaining_time:.2f} seconds.")
        yield future.result()

import csv
import time
import numpy as np

# First try to load existing results from CSV
commands_with_descriptions = []
try:
    with open('commands_with_descriptions.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        commands_with_descriptions = list((row[0], row[1]) for row in reader)
    print(f"Loaded {len(commands_with_descriptions)} existing command descriptions from CSV")
except FileNotFoundError:
    # If CSV doesn't exist, compute the descriptions
    print("No existing results found, computing descriptions...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_command, command) for command in commands]
        commands_with_descriptions = list(track_progress(futures, len(commands)))
        
        # Save results to CSV
        with open('commands_with_descriptions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Command', 'Description'])  # Write header
            for cmd, desc in commands_with_descriptions:
                try:
                    writer.writerow([cmd, desc])
                except Exception as e:
                    print(f"Failed to write row for command {cmd}: {e}")
                    continue

## Use sentences transformers to calculate embeddings 
model = SentenceTransformer('all-MiniLM-L6-v2')

# First try to load existing embeddings from CSV
cmd_coordinates = []
try:
    embeddings_file = 'command_embeddings.csv'
    loaded_embeddings = np.loadtxt(embeddings_file, delimiter=',')
    for i, cmd_with_description in enumerate(commands_with_descriptions):
        cmd_coordinates.append((cmd_with_description[0], loaded_embeddings[i]))
    print(f"Loaded {len(cmd_coordinates)} existing embeddings from CSV")
except:
    print("Computing embeddings...")
    start_time = time.time()
    total = len(commands_with_descriptions)
    
    for i, cmd_with_description in enumerate(commands_with_descriptions):
        sentence = cmd_with_description[0]+":"+cmd_with_description[1]
        embedding = model.encode(sentence)
        cmd_coordinates.append((cmd_with_description[0], embedding))
        
        # Progress tracking
        elapsed = time.time() - start_time
        progress = (i + 1) / total
        eta = elapsed / progress * (1 - progress)
        print(f"Progress: {i+1}/{total} ({progress:.1%}) - ETA: {eta:.1f}s", end='\r')

    print("\nSaving embeddings to CSV...")
    embeddings = np.array([coord[1] for coord in cmd_coordinates])
    np.savetxt('command_embeddings.csv', embeddings, delimiter=',')

#Generate coordinate for text provided
embedding_text = model.encode(text_to_use_as_input)

## Deduce the 50 nearest command lines 
cos_sim_value = []

for cmd_coordinate in cmd_coordinates:
    cos_sim = np.dot(embedding_text, cmd_coordinate[1])/(np.linalg.norm(embedding_text)*np.linalg.norm(cmd_coordinate[1]))
    cos_sim_value.append((cmd_coordinate[0], cos_sim))

sorted_list = sorted(cos_sim_value, key=lambda x: x[1], reverse=True)

top_50 = sorted_list[:50]

print(top_50)
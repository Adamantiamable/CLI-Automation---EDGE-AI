import os
import subprocess
from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer, util

# Text Provided

text_to_use_as_input = "I want to find my LLM directory "

# Get AI text output
def get_AI_output(prompt, max_retries=3, retry_delay=2):
     for attempt in range(max_retries):
        try : 
            client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key = "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",
            )
            
            completion = client.chat.completions.create(
            extra_headers={},
            model="liquid/lfm-40b:free",
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

for command in commands:
    prompt = f"Take this CLI command {command}, provide a description of it and its associated arguments"
    description = get_AI_output(prompt)
    print(f"Prompt: {prompt!r}, Generated text: {description!r}")
    commands_with_descriptions += (command, description)

## Use sentences transformers to calculate embeddings 
model = SentenceTransformer('all-MiniLM-L6-v2')


#Generates coordonates for commands 
cmd_coordinates = []
for cmd_with_description in commands_with_descriptions:
    sentence = cmd_with_description[0]+":"+cmd_with_description[1];
    embedding = model.encode(sentence)
    cmd_coordinates += (cmd, embedding)

#Generate coordonate for text provided
embedding_text = model.encode(text_to_use_as_input);

## Deduce the 50 nearest command lines 

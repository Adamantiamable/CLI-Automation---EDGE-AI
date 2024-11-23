import warnings

import concurrent
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import csv
from typing import List, Set, Tuple
import dotenv
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
import time
from tqdm import tqdm

dotenv.load_dotenv()
# API_KEY = os.getenv("OPENAI_API_KEY")
from constants import API_KEY

# Define command templates and their metadata
WINDOWS_COMMANDS = {
    'copy': {
        'description': 'Copy one or more files from source to destination',
        'args': ['source', 'destination'],
        'template': lambda paths: f'copy "{paths[0]}" "{paths[1]}"'
    },
    'move': {
        'description': 'Move or rename files from source to destination',
        'args': ['source', 'destination'],
        'template': lambda paths: f'move "{paths[0]}" "{paths[1]}"'
    },
    'del': {
        'description': 'Delete one or more files',
        'args': ['target'],
        'template': lambda paths: f'del "{paths[0]}"'
    },
    'dir': {
        'description': 'List files and folders in a directory',
        'args': ['directory'],
        'template': lambda paths: f'dir "{paths[0]}"'
    },
    'type': {
        'description': 'Display the contents of a text file',
        'args': ['file'],
        'template': lambda paths: f'type "{paths[0]}"'
    },
    'code': {
        'description': 'Open file or directory in Visual Studio Code',
        'args': ['target'],
        'template': lambda paths: f'code "{paths[0]}"'
    },
    'notepad': {
        'description': 'Open file in Notepad text editor',
        'args': ['file'],
        'template': lambda paths: f'notepad "{paths[0]}"'
    },
    'explorer': {
        'description': 'Open File Explorer in specified directory',
        'args': ['directory'],
        'template': lambda paths: f'explorer "{paths[0]}"'
    },
    'mkdir': {
        'description': 'Create a new directory',
        'args': ['directory'],
        'template': lambda paths: f'mkdir "{paths[0]}"'
    },
    'rmdir': {
        'description': 'Remove an empty directory',
        'args': ['directory'],
        'template': lambda paths: f'rmdir "{paths[0]}"'
    }
}

def generate_command_embeddings(cache_file="command_embeddings.csv"):
    """Generate and cache embeddings for Windows commands"""
    if os.path.exists(cache_file):
        print("Loading existing command embeddings...")
        df = pd.read_csv(cache_file)
        return {row['command']: eval(row['embedding']) for _, row in df.iterrows()}
    
    print("Generating command embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = {}
    for cmd, metadata in WINDOWS_COMMANDS.items():
        embedding = model.encode(metadata['description'])
        embeddings[cmd] = embedding
    
    # Save embeddings
    with open(cache_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['command', 'description', 'embedding'])
        for cmd, embedding in embeddings.items():
            writer.writerow([cmd, WINDOWS_COMMANDS[cmd]['description'], embedding.tolist()])
    
    return embeddings

def find_nearest_command(query: str, descriptions, file_embeddings):
    """Find the nearest Windows command and generate appropriate file paths based on the query"""
    # Load or generate command embeddings
    command_embeddings = generate_command_embeddings()
    
    # Find closest command
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    
    # Calculate command similarities
    cmd_similarities = {}
    for cmd, embedding in command_embeddings.items():
        similarity = 1 - cosine(query_embedding, embedding)
        cmd_similarities[cmd] = similarity
    
    # Get best matching command
    best_command = max(cmd_similarities.items(), key=lambda x: x[1])[0]
    command_metadata = WINDOWS_COMMANDS[best_command]
    
    # Find paths for each argument
    selected_paths = []
    for arg_name in command_metadata['args']:
        # Modify query to focus on specific argument
        arg_query = f"{arg_name} for {query}"
        path, desc, score = find_closest_path(arg_query, "file_embeddings.csv")
        selected_paths.append(path)
    
    # Generate final command using template
    final_command = command_metadata['template'](selected_paths)
    
    return {
        'command': best_command,
        'description': command_metadata['description'],
        'arguments': dict(zip(command_metadata['args'], selected_paths)),
        'generated_command': final_command,
        'confidence': cmd_similarities[best_command]
    }

def walk_directory(workdir: str, exclude_dirs: Set[str] = {'.git', 'node_modules', '__pycache__', '.svn'}) -> Tuple[List[str], List[List[int]], List[List[int]]]:
    """
    Walk through directory structure collecting paths and sibling relationships.
    
    Args:
        workdir: Starting directory path
        exclude_dirs: Set of directory names to skip
        
    Returns:
        Tuple containing:
        - List of all paths (relative to workdir)
        - List of lists containing sibling path indices for each path
        - List of lists containing direct child file indices for each directory
    """
    all_paths = []
    sibling_indices = []
    dir_file_indices = []
    
    # Convert workdir to absolute path for consistent handling
    workdir = os.path.abspath(workdir)
    
    def normalize_path(path: str) -> str:
        """Normalize path to use forward slashes and be relative to workdir"""
        if path == workdir:
            return '.'
        rel_path = os.path.relpath(path, workdir)
        # Convert Windows backslashes to forward slashes
        return rel_path.replace(os.sep, '/')
    
    # First pass: collect all paths and build directory structure
    dir_contents = {}  # Map of directory to its immediate contents (files and dirs)
    
    for root, dirs, files in os.walk(workdir, topdown=True):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        dirs.sort()  # Sort for consistent ordering
        files.sort()
        
        current_dir = normalize_path(root)
        
        # Add current directory if not already added
        if current_dir not in all_paths:
            all_paths.append(current_dir)
        
        # Store directory contents
        dir_contents[current_dir] = {
            'dirs': [normalize_path(os.path.join(root, d)) for d in dirs],
            'files': [normalize_path(os.path.join(root, f)) for f in files]
        }
        
        # Add all files
        for file in files:
            file_path = normalize_path(os.path.join(root, file))
            if file_path not in all_paths:
                all_paths.append(file_path)
    
    # Second pass: build relationships
    for path in all_paths:
        if path == '.' or os.path.isdir(os.path.join(workdir, path)):
            # Handle directories (including root)
            if path == '.':
                # Root directory has no siblings
                sibling_indices.append([])
            else:
                # Get parent directory and find sibling directories
                parent_dir = normalize_path(os.path.dirname(os.path.join(workdir, path)))
                siblings = []
                if parent_dir in dir_contents:
                    for sibling in dir_contents[parent_dir]['dirs']:
                        if sibling != path:
                            sibling_idx = all_paths.index(sibling)
                            siblings.append(sibling_idx)
                sibling_indices.append(siblings)
            
            # Get child files
            child_files = []
            if path in dir_contents:
                for file in dir_contents[path]['files']:
                    file_idx = all_paths.index(file)
                    child_files.append(file_idx)
            dir_file_indices.append(child_files)
            
        else:
            # Handle files
            dir_path = normalize_path(os.path.dirname(os.path.join(workdir, path)))
            siblings = []
            if dir_path in dir_contents:
                for sibling in dir_contents[dir_path]['files']:
                    if sibling != path:
                        sibling_idx = all_paths.index(sibling)
                        siblings.append(sibling_idx)
            sibling_indices.append(siblings)
            dir_file_indices.append([])  # Files don't have child files
    
    return all_paths, sibling_indices, dir_file_indices


def generate_file_descriptions(max_depth=3, cache_file="file_descriptions.csv", embeddings_file="file_embeddings.csv"):
    import concurrent.futures
    
    # Check if both files exist and have data
    if os.path.exists(cache_file) and os.path.exists(embeddings_file):
        print("Found existing description and embedding files.")
        # Read and return existing data
        with open(cache_file, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
            descriptions = dict(zip(df['path'], df['description']))
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
            embeddings = {row['path']: eval(row['embedding']) for _, row in df.iterrows()}
        print(f"Loaded {len(descriptions)} existing descriptions and embeddings.")
        return descriptions, embeddings
    
    # Initialize sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key = API_KEY,
    )
    
    # Load existing descriptions if available
    descriptions = {}
    if os.path.exists(cache_file):
        print("Loading existing descriptions...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            descriptions = {row[0]: row[1] for row in reader}
    
    # Get all files and directories
    paths, siblings, children = walk_directory(os.getcwd())
    
    # Filter paths to only process new ones
    paths_to_process = [p for p in paths if p not in descriptions]
    
    if not paths_to_process:
        print("All paths already have descriptions. Moving to embeddings...")
    else:
        # Rest of the get_description function and processing remains the same
        def get_description(path_index):
            path = paths[path_index]
            sibling_files = [paths[s] for s in siblings[path_index]]
            child_files = [paths[c] for c in children[path_index]]

            files_info = ""
            if len(sibling_files) > 0:
                files_info = f"\nFiles in same directory: {', '.join(sibling_files)}"
            if len(child_files) > 0:
                files_info += f"\nFiles directly underneath: {', '.join(child_files)}"

            if files_info != "":
                files_info = f"\n{files_info}"
            
            prompt = f"Given the file path and the file context, describe in one short sentence what might be in this file or directory: {path}\n{files_info}"
            
            try:
                response = client.chat.completions.create(
                    model="google/gemini-flash-1.5-8b",
                    messages=[{"role": "user", "content": prompt}]
                )
                return path, f"File {path}: {response.choices[0].message.content[:500]}"
            except Exception as e:
                print(f"Error processing {path}: {e}")
                return path, "Error: Could not generate description"

        # Process only new paths
        start_time = time.time()
        print('Process new path...')
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_description, i) for i, p in enumerate(paths) if p in paths_to_process]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                                total=len(paths_to_process), 
                                desc="Generating descriptions"):
                path, desc = future.result()
                descriptions[path] = desc

    print("\nGenerating embeddings...")
    # Generate embeddings only for paths that don't have them
    embeddings = {}
    for path, desc in tqdm(descriptions.items(), desc="Computing embeddings"):
        embedding = model.encode(desc)
        embeddings[path] = embedding
    
    # Save all results
    print("Saving results to CSV...")
    with open(cache_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'description'])
        for path, desc in descriptions.items():
            writer.writerow([path, desc])
    
    with open(embeddings_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'description', 'embedding'])
        for path, desc in descriptions.items():
            writer.writerow([path, desc, embeddings[path].tolist()])
            
    return descriptions, embeddings

def determine_query_type(query):
    """Determine if query is looking for a file or directory"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    file_reference = "looking for a file"
    dir_reference = "looking for a directory"
    
    # Get embeddings
    query_emb = model.encode(query)
    file_emb = model.encode(file_reference)
    dir_emb = model.encode(dir_reference)
    
    # Calculate similarities
    file_sim = 1 - cosine(query_emb, file_emb)
    dir_sim = 1 - cosine(query_emb, dir_emb)
    
    return 'file' if file_sim > dir_sim else 'directory'

def find_closest_path(query, embeddings_file="file_embeddings.csv"):
    # Determine query type
    query_type = determine_query_type(query)
    
    # Load embeddings
    df = pd.read_csv(embeddings_file)
    
    # Filter paths based on query type
    is_file = df['path'].apply(lambda x: '.' in os.path.basename(x) if x != '.' else False)
    if query_type == 'file':
        df = df[is_file]
    else:
        df = df[~is_file]
    
    paths = df['path'].tolist()
    descriptions = df['description'].tolist()
    embeddings = [eval(e) for e in df['embedding']]
    
    # Generate query embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)

    print("Calculating similarities...")
    # Calculate similarities
    similarities = []
    for emb in embeddings:
        similarity = 1 - cosine(query_embedding, emb)
        similarities.append(similarity)
    
    # Find best match
    best_idx = np.argmax(similarities)
    return paths[best_idx], descriptions[best_idx], similarities[best_idx]

# Example usage
if __name__ == "__main__":
    descriptions, embeddings = generate_file_descriptions()

    # Then find nearest command
    # query = "I want to look at the contents of my inference.py file"
    query = "I want to move my extraction file python file to the destination output folder"
    result = find_nearest_command(query, descriptions, embeddings)

    print(f"Query: {query}")
    print(f"Command: {result['command']}")
    print(f"Description: {result['description']}")
    print(f"Arguments: {result['arguments']}")
    print(f"Generated command: {result['generated_command']}")
    print(f"Confidence: {result['confidence']:.3f}")
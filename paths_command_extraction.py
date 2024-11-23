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
API_KEY = os.getenv("OPENAI_API_KEY")


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
            
            prompt = f"Given the file path and the file context, describe very quickly what might be in this file or directory: {path}\n{files_info}"
            
            try:
                response = client.chat.completions.create(
                    model="google/gemini-flash-1.5-8b",
                    messages=[{"role": "user", "content": prompt}]
                )
                return path, response.choices[0].message.content
            except Exception as e:
                print(f"Error processing {path}: {e}")
                return path, "Error: Could not generate description"

        # Process only new paths
        start_time = time.time()
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

    print("Finding closest extensions...")
    closest_ext = find_closest_extensions(query, query_embedding)
    
    print("Calculating similarities...")
    # Calculate similarities
    similarities = []
    for emb in embeddings:
        similarity = 1 - cosine(query_embedding, emb)
        if is_file[embeddings.index(emb)]:
            similarity = adjust_similarity_score(paths[embeddings.index(emb)], similarity, query, closest_ext)
        similarities.append(similarity)
    
    # Find best match
    best_idx = np.argmax(similarities)
    return paths[best_idx], descriptions[best_idx], similarities[best_idx]

def get_file_extensions():
    return {
        # Documents and Text
        '.txt': 'Plain text document',
        '.doc': 'Microsoft Word document',
        '.docx': 'Microsoft Word document (XML-based)',
        '.pdf': 'Portable Document Format',
        '.rtf': 'Rich Text Format',
        '.odt': 'OpenDocument text document',
        '.md': 'Markdown document',
        '.tex': 'LaTeX document',
        '.wpd': 'WordPerfect document',
        '.pages': 'Apple Pages document',

        # Spreadsheets and Data
        '.xlsx': 'Microsoft Excel spreadsheet (XML-based)',
        '.xls': 'Microsoft Excel spreadsheet',
        '.csv': 'Comma-separated values',
        '.ods': 'OpenDocument spreadsheet',
        '.tsv': 'Tab-separated values',
        '.numbers': 'Apple Numbers spreadsheet',

        # Presentations
        '.ppt': 'Microsoft PowerPoint presentation',
        '.pptx': 'Microsoft PowerPoint presentation (XML-based)',
        '.key': 'Apple Keynote presentation',
        '.odp': 'OpenDocument presentation',

        # Images
        '.jpg': 'JPEG image',
        '.jpeg': 'JPEG image',
        '.png': 'Portable Network Graphics image',
        '.gif': 'Graphics Interchange Format image',
        '.bmp': 'Bitmap image',
        '.tiff': 'Tagged Image File Format',
        '.svg': 'Scalable Vector Graphics',
        '.webp': 'WebP image',
        '.raw': 'Raw image format',
        '.psd': 'Adobe Photoshop document',
        '.ai': 'Adobe Illustrator document',

        # Audio
        '.mp3': 'MPEG Layer 3 audio',
        '.wav': 'Waveform audio',
        '.ogg': 'Ogg Vorbis audio',
        '.flac': 'Free Lossless Audio Codec',
        '.m4a': 'MPEG-4 audio',
        '.aac': 'Advanced Audio Coding',
        '.wma': 'Windows Media Audio',
        '.mid': 'MIDI audio',
        '.aiff': 'Audio Interchange File Format',

        # Video
        '.mp4': 'MPEG-4 video',
        '.avi': 'Audio Video Interleave',
        '.mov': 'Apple QuickTime movie',
        '.wmv': 'Windows Media Video',
        '.flv': 'Flash Video',
        '.mkv': 'Matroska Video',
        '.webm': 'WebM video',
        '.m4v': 'MPEG-4 video',
        '.3gp': '3GPP multimedia container',

        # Programming and Development
        '.py': 'Python source code',
        '.java': 'Java source code',
        '.class': 'Java compiled class',
        '.js': 'JavaScript source code',
        '.jsx': 'JavaScript React',
        '.ts': 'TypeScript source code',
        '.tsx': 'TypeScript React',
        '.html': 'HyperText Markup Language',
        '.htm': 'HyperText Markup Language',
        '.css': 'Cascading Style Sheets',
        '.scss': 'Sass stylesheet',
        '.less': 'Less stylesheet',
        '.php': 'PHP source code',
        '.c': 'C source code',
        '.cpp': 'C++ source code',
        '.h': 'C/C++ header file',
        '.cs': 'C# source code',
        '.rb': 'Ruby source code',
        '.go': 'Go source code',
        '.rs': 'Rust source code',
        '.swift': 'Swift source code',
        '.kt': 'Kotlin source code',
        '.sql': 'SQL database file',

        # Compressed and Archive
        '.zip': 'ZIP archive',
        '.rar': 'RAR archive',
        '.7z': '7-Zip archive',
        '.tar': 'Tape archive',
        '.gz': 'Gzip compressed file',
        '.bz2': 'Bzip2 compressed file',

        # System and Configuration
        '.exe': 'Windows executable',
        '.dll': 'Dynamic Link Library',
        '.sys': 'System file',
        '.ini': 'Configuration file',
        '.cfg': 'Configuration file',
        '.xml': 'Extensible Markup Language',
        '.json': 'JavaScript Object Notation',
        '.yaml': 'YAML configuration file',
        '.yml': 'YAML configuration file',
        '.log': 'Log file',
        '.bat': 'Windows batch file',
        '.sh': 'Shell script',

        # Database
        '.db': 'Database file',
        '.sqlite': 'SQLite database',
        '.mdb': 'Microsoft Access database',
        '.accdb': 'Microsoft Access database',
        '.dbf': 'dBase database',

        # Font
        '.ttf': 'TrueType font',
        '.otf': 'OpenType font',
        '.woff': 'Web Open Font Format',
        '.woff2': 'Web Open Font Format 2',
        '.eot': 'Embedded OpenType font',

        # Email and Calendar
        '.eml': 'Email message',
        '.msg': 'Outlook email message',
        '.ics': 'iCalendar file',
        '.vcf': 'vCard contact file',

        # 3D and CAD
        '.obj': '3D object file',
        '.stl': 'Stereolithography file',
        '.fbx': 'Filmbox 3D file',
        '.blend': 'Blender 3D file',
        '.dae': 'COLLADA 3D file',
        '.dwg': 'AutoCAD drawing',
        '.skp': 'SketchUp file',

        # Game Development
        '.unity': 'Unity scene file',
        '.unitypackage': 'Unity asset package',
        '.prefab': 'Unity prefab file',
        '.uasset': 'Unreal Engine asset',
        '.map': 'Game map file',
        '.bsp': 'Binary Space Partition'
    }

def get_extension_similarity(query, query_emb, extension_desc, embeddings_cache="extension_embeddings.csv"):
    """Calculate similarity between query and file extension description using cached embeddings"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load or create extension embeddings cache
    if os.path.exists(embeddings_cache):
        print("Loading extension embeddings cache...")
        df = pd.read_csv(embeddings_cache)
        ext_emb = np.array(eval(df[df['description'] == extension_desc]['embedding'].iloc[0]))
    else:
        # Generate embeddings for all extensions
        extensions = get_file_extensions()
        embeddings_data = []
        
        print("Generating extension embeddings cache...")
        for ext, desc in extensions.items():
            emb = model.encode(desc).tolist()
            embeddings_data.append([ext, desc, emb])
            
        # Save embeddings cache
        df = pd.DataFrame(embeddings_data, columns=['extension', 'description', 'embedding'])
        df.to_csv(embeddings_cache, index=False)
        ext_emb = model.encode(extension_desc)
    
    return 1 - cosine(query_emb, ext_emb)

def find_closest_extensions(query, query_emb):
    """Find top 5 extensions most relevant to the query using parallel processing"""
    extensions = get_file_extensions()
    
    # Process extension similarities in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ext = {
            executor.submit(get_extension_similarity, query, query_emb, desc): ext 
            for ext, desc in extensions.items()
        }
        
        similarities = []
        for future in concurrent.futures.as_completed(future_to_ext):
            ext = future_to_ext[future]
            try:
                similarity = future.result()
                similarities.append((ext, similarity))
            except Exception as e:
                print(f"Error processing {ext}: {e}")
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

def adjust_similarity_score(path, score, query, top_extensions):
    """Adjust similarity score based on file extension relevance"""
    if '.' not in path or path == '.':
        return score
        
    extension = '.' + path.split('.')[-1]
    
    # Give bonus if file extension is among top 5 relevant extensions
    for i, (ext, ext_score) in enumerate(top_extensions):
        if extension == ext:
            # Bonus decreases with position in top 5
            bonus = 0.1 * (5-i) / 5
            return score + bonus
            
    return score

# Example usage
if __name__ == "__main__":
    # First run to generate descriptions and embeddings
    descriptions, embeddings = generate_file_descriptions(max_depth=3)
    
    # Then search
    query = "the txt file where I stored the commands I got"
    path, desc, score = find_closest_path(query)
    print(f"Query: {query}")
    print(f"Best match: {path}")
    print(f"Description: {desc}")
    print(f"Similarity score: {score:.3f}")
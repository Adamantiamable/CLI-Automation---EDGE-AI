import os
from typing import List, Tuple, Set

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

# Example usage demonstration
if __name__ == "__main__":
    paths, siblings, dir_files = walk_directory(".")
    
    print("All paths:")
    for i, path in enumerate(paths):
        print(f"{i}: {path}")
    
    print("\nSibling relationships:")
    for i, sibs in enumerate(siblings):
        if sibs:  # Only print non-empty sibling lists
            print(f"{paths[i]} has siblings: {[paths[j] for j in sibs]}")
    
    print("\nDirectory files:")
    for i, files in enumerate(dir_files):
        if files:  # Only print non-empty file lists
            print(f"{paths[i]} contains files: {[paths[j] for j in files]}")
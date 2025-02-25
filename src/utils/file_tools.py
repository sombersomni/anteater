import glob
import os
from typing import Callable

def count_matching_files(folder_name, query, match_type="extension"):
    """
    Count files in a folder that match a query.
    
    Parameters:
    - folder_name (str): Path to the folder
    - query (str): String to match against (e.g., file extension, name substring)
    - match_type (str): Type of match ('extension', 'name_contains', 'exact_name')
    
    Returns:
    - int: Number of matching files
    """
    return len(search_files(folder_name, query, match_type))

def search_files(folder_path, pattern="*.*", recursive=False):
    """
    Searches for files in a specified folder matching a given pattern.
    
    Args:
        folder_path (str): Path to the folder (e.g., "C:/my_folder" or "./my_folder")
        pattern (str): File pattern to match (e.g., "*.txt" for text files, "*.*" for all files)
        recursive (bool): If True, searches subdirectories recursively

    Returns:
        list: List of file paths matching the pattern
    """
    # Check if folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Normalize path separator for cross-platform compatibility
    folder_path = os.path.normpath(folder_path)
    # Build the search pattern
    search_path = os.path.join(folder_path, pattern)
    if recursive:
        search_path = os.path.join(folder_path, "**", pattern)
    
    # Get list of files matching pattern
    return glob.glob(search_path, recursive=recursive)

def apply_action_to_files(
    folder_path: str,
    pattern: str,
    file_action: Callable[[str], None],
    recursive: bool=False
):
    """
    Removes files in a specified folder matching a given pattern.
    
    Args:
        folder_path (str): Path to the folder (e.g., "C:/my_folder" or "./my_folder")
        pattern (str): File pattern to match (e.g., "*.txt" for text files, "*.*" for all files)
        recursive (bool): If True, searches subdirectories recursively
    
    Returns:
        int: Number of files acted upon
    
    Raises:
        FileNotFoundError: If the folder doesn't exist
        PermissionError: If access to folder/files is denied
    """
    try:
        # Check if folder exists
        files_found = search_files(folder_path, pattern, recursive)
        # Counter for removed files
        count = 0
        # Remove each file
        for file_path in files_found:
            if os.path.isfile(file_path):  # Ensure it's a file, not a directory
                file_action(file_path)
                count += 1        
        return count
    
    except PermissionError as e:
        print(f"Permission denied: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

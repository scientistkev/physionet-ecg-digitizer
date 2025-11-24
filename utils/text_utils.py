"""
Text file utilities for loading, saving, and parsing text files.
"""

from typing import Tuple, List, Union
from pathlib import Path


def load_text(filename: Union[str, Path]) -> str:
    """
    Load a text file as a string.
    
    Args:
        filename: Path to the text file to load.
        
    Returns:
        Contents of the file as a string.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def save_text(filename: Union[str, Path], string: str) -> None:
    """
    Save a string as a text file.
    
    Args:
        filename: Path where the text file should be saved.
        string: String content to write to the file.
        
    Raises:
        IOError: If the file cannot be written.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(string)


def get_variable(string: str, variable_name: str) -> Tuple[str, bool]:
    """
    Get a variable from a string by searching for lines starting with variable_name.
    
    Args:
        string: Multi-line string to search.
        variable_name: Prefix to search for at the start of lines.
        
    Returns:
        Tuple of (variable_value, has_variable) where variable_value is the
        stripped content after the variable_name prefix, and has_variable
        indicates whether a matching line was found.
    """
    lines = string.split('\n')
    for line in lines:
        if line.startswith(variable_name):
            return line[len(variable_name):].strip(), True
    return '', False


def get_variables(string: str, variable_name: str, sep: str = ',') -> Tuple[List[str], bool]:
    """
    Get variables from a string by searching for lines starting with variable_name
    and splitting by the separator.
    
    Args:
        string: Multi-line string to search.
        variable_name: Prefix to search for at the start of lines.
        sep: Separator used to split multiple values. Default is ','.
        
    Returns:
        Tuple of (variables, has_variable) where variables is a list of stripped
        variable values, and has_variable indicates whether a matching line was found.
    """
    lines = string.split('\n')
    for line in lines:
        if line.startswith(variable_name):
            suffix = line[len(variable_name):].strip()
            variables = [var.strip() for var in suffix.split(sep) if var.strip()]
            return variables, True
    return [], False


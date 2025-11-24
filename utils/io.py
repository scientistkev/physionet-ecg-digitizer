"""
Data I/O operations for loading and saving records, signals, images, and labels.

This module provides functions for reading and writing WFDB records, including
headers, signals, images, and labels used in the PhysioNet ECG digitization challenge.
"""

from typing import List, Tuple, Optional, Union, Any, Set
from pathlib import Path
import os
import wfdb
from PIL import Image
from . import constants
from . import text_utils
from . import header_parser


def find_records(folder: Union[str, Path]) -> List[str]:
    """
    Find all WFDB records in a folder and its subfolders.
    
    Searches recursively for files with .hea extension and extracts the record
    names (without extension).
    
    Args:
        folder: Root directory to search for records.
        
    Returns:
        Sorted list of record names (relative paths without .hea extension).
    """
    folder_path = Path(folder)
    records: Set[str] = set()
    
    for hea_file in folder_path.rglob('*.hea'):
        # Get relative path from folder, remove .hea extension
        record = str(hea_file.relative_to(folder_path))[:-4]
        records.add(record)
    
    return sorted(records)


def load_header(record: Union[str, Path]) -> str:
    """
    Load the header for a record.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        Header file content as a string.
        
    Raises:
        FileNotFoundError: If the header file does not exist.
    """
    header_file = header_parser.get_header_file(record)
    return text_utils.load_text(header_file)


def load_signals(
    record: Union[str, Path]
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load the signals for a record.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        Tuple of (signal, fields) where:
        - signal: Signal array from wfdb.rdsamp, or None if signals don't exist
        - fields: Signal metadata dictionary from wfdb.rdsamp, or None if signals don't exist
    """
    signal_files = header_parser.get_signal_files(record)
    record_path = Path(record)
    record_dir = record_path.parent if record_path.parent != Path('.') else Path('.')
    
    # Check if all signal files exist
    signal_files_exist = all(
        (record_dir / signal_file).is_file() for signal_file in signal_files
    )
    
    if signal_files and signal_files_exist:
        signal, fields = wfdb.rdsamp(str(record))
        return signal, fields
    else:
        return None, None


def load_images(record: Union[str, Path]) -> List[Image.Image]:
    """
    Load the images for a record.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        List of PIL Image objects for all images referenced in the header.
    """
    record_path = Path(record)
    record_dir = record_path.parent if record_path.parent != Path('.') else Path('.')
    image_files = header_parser.get_image_files(record)

    images: List[Image.Image] = []
    for image_file in image_files:
        image_file_path = record_dir / image_file
        if image_file_path.is_file():
            image = Image.open(image_file_path)
            images.append(image)

    return images


def load_labels(record: Union[str, Path]) -> List[str]:
    """
    Load the labels for a record.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        List of labels found in the header.
        
    Raises:
        ValueError: If no labels are found in the header.
    """
    header = load_header(record)
    return header_parser.get_labels_from_header(header)


def save_header(record: Union[str, Path], header: str) -> None:
    """
    Save the header for a record.
    
    Args:
        record: Record path (with or without .hea extension).
        header: Header content to write.
        
    Raises:
        IOError: If the file cannot be written.
    """
    header_file = header_parser.get_header_file(record)
    text_utils.save_text(header_file, header)


def save_signals(
    record: Union[str, Path],
    signal: Any,
    comments: Optional[List[str]] = None
) -> None:
    """
    Save the signals for a record.
    
    Reads the existing header to extract metadata, then writes the signal
    data using wfdb.wrsamp.
    
    Args:
        record: Record path (with or without .hea extension).
        signal: Signal array to save (compatible with wfdb format).
        comments: Optional list of comment strings to add to the header.
                 '#' characters are automatically stripped.
                 
    Raises:
        FileNotFoundError: If the header file does not exist.
        IOError: If the signal files cannot be written.
    """
    if comments is None:
        comments = []
    
    header = load_header(record)
    record_path = Path(record)
    record_name = record_path.name
    record_dir = str(record_path.parent) if record_path.parent != Path('.') else '.'
    
    # Extract metadata from header
    sampling_frequency = header_parser.get_sampling_frequency(header)
    signal_formats = header_parser.get_signal_formats(header)
    adc_gains = header_parser.get_adc_gains(header)
    baselines = header_parser.get_baselines(header)
    signal_units = header_parser.get_signal_units(header)
    signal_names = header_parser.get_signal_names(header)
    
    # Clean comments (remove '#' and strip whitespace)
    cleaned_comments = [comment.replace('#', '').strip() for comment in comments]

    wfdb.wrsamp(
        record_name,
        fs=sampling_frequency,
        units=signal_units,
        sig_name=signal_names,
        p_signal=signal,
        fmt=signal_formats,
        adc_gain=adc_gains,
        baseline=baselines,
        comments=cleaned_comments,
        write_dir=record_dir
    )


def save_labels(record: Union[str, Path], labels: List[str]) -> str:
    """
    Save the labels for a record by appending them to the header file.
    
    Args:
        record: Record path (with or without .hea extension).
        labels: List of label strings to save.
        
    Returns:
        Updated header content as a string.
        
    Raises:
        FileNotFoundError: If the header file does not exist.
        IOError: If the header file cannot be written.
    """
    header_file = header_parser.get_header_file(record)
    header = text_utils.load_text(header_file)
    header += constants.substring_labels + ' ' + ', '.join(labels) + '\n'
    text_utils.save_text(header_file, header)
    return header

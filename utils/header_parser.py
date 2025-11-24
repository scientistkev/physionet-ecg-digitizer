"""
Header parsing functions for challenge-specific and WFDB header files.

This module provides functions to parse WFDB header files and extract
metadata about signals, images, labels, and signal properties.
"""

from typing import List, Optional, Union
from pathlib import Path
import os
from . import constants
from . import text_utils
from . import helpers


def get_signal_files_from_header(header: str) -> List[str]:
    """
    Get the signal files from a header or a similar string.
    
    Parses the header to extract signal file names from the signal definition
    lines. The first non-comment line contains the number of signals.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of unique signal file names found in the header.
    """
    lines = header.split('\n')
    signal_files: List[str] = []
    num_channels: Optional[int] = None
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        if i == 0 and len(parts) > 1:
            # First line: record_name num_signals sampling_freq num_samples
            try:
                num_channels = int(parts[1])
            except (ValueError, IndexError):
                break
        elif num_channels is not None and 1 <= i <= num_channels:
            # Signal definition lines
            signal_file = parts[0]
            if signal_file and signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
            
    return signal_files


def get_image_files_from_header(header: str) -> List[str]:
    """
    Get the image files from a header or a similar string.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of image file names found in the header.
        
    Raises:
        ValueError: If no images are found in the header.
    """
    images, has_image = text_utils.get_variables(header, constants.substring_images)
    if not has_image:
        raise ValueError(
            'No images available: did you forget to generate or include the images?'
        )
    return images


def get_labels_from_header(header: str) -> List[str]:
    """
    Get the labels from a header or a similar string.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of labels found in the header.
        
    Raises:
        ValueError: If no labels are found in the header.
    """
    labels, has_labels = text_utils.get_variables(header, constants.substring_labels)
    if not has_labels:
        raise ValueError(
            'No labels available: are you trying to load the labels from the '
            'held-out data, or did you forget to prepare the data to include the labels?'
        )
    return labels


def get_header_file(record: Union[str, Path]) -> str:
    """
    Get the header file path for a record.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        Header file path with .hea extension.
    """
    record_str = str(record)
    if not record_str.endswith('.hea'):
        return record_str + '.hea'
    return record_str


def get_signal_files(record: Union[str, Path]) -> List[str]:
    """
    Get the signal files for a record by reading its header.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        List of signal file names.
    """
    header_file = get_header_file(record)
    header = text_utils.load_text(header_file)
    return get_signal_files_from_header(header)


def get_image_files(record: Union[str, Path]) -> List[str]:
    """
    Get the image files for a record by reading its header.
    
    Args:
        record: Record path (with or without .hea extension).
        
    Returns:
        List of image file names.
    """
    header_file = get_header_file(record)
    header = text_utils.load_text(header_file)
    return get_image_files_from_header(header)


# WFDB functions

def _parse_header_first_line(header: str) -> List[str]:
    """
    Parse the first line of a WFDB header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of fields from the first line.
    """
    first_line = header.split('\n')[0]
    return first_line.split()


def get_record_name(header: str) -> str:
    """
    Get the record name from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        Record name (first field of first line, before any '/').
    """
    fields = _parse_header_first_line(header)
    if fields:
        return fields[0].split('/')[0].strip()
    return ''


def get_num_signals(header: str) -> Optional[int]:
    """
    Get the number of signals from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        Number of signals, or None if not found or invalid.
    """
    fields = _parse_header_first_line(header)
    if len(fields) > 1:
        value = fields[1].strip()
        if helpers.is_integer(value):
            return int(value)
    return None


def get_sampling_frequency(header: str) -> Optional[float]:
    """
    Get the sampling frequency from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        Sampling frequency in Hz, or None if not found or invalid.
    """
    fields = _parse_header_first_line(header)
    if len(fields) > 2:
        value = fields[2].split('/')[0].strip()
        if helpers.is_number(value):
            return float(value)
    return None


def get_num_samples(header: str) -> Optional[int]:
    """
    Get the number of samples from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        Number of samples, or None if not found or invalid.
    """
    fields = _parse_header_first_line(header)
    if len(fields) > 3:
        value = fields[3].strip()
        if helpers.is_integer(value):
            return int(value)
    return None


def get_signal_formats(header: str) -> List[str]:
    """
    Get the signal formats from a header file.
    
    Extracts the format field from each signal definition line, handling
    various format specifications (e.g., '16', '16x', '16:8', '16+8').
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of signal format strings, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[str] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 1:
            field = parts[1]
            # Handle format specifications: remove 'x', ':', '+' suffixes
            for sep in ['x', ':', '+']:
                if sep in field:
                    field = field.split(sep)[0]
            values.append(field)
    
    return values


def get_adc_gains(header: str) -> List[float]:
    """
    Get the ADC gains from a header file.
    
    Extracts the gain value from each signal definition line, handling
    formats like 'gain/unit' or 'gain(baseline)'.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of ADC gain values, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[float] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 2:
            field = parts[2]
            # Remove unit and baseline: 'gain/unit' or 'gain(baseline)'
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[0]
            try:
                values.append(float(field))
            except ValueError:
                values.append(0.0)
    
    return values


def get_baselines(header: str) -> List[int]:
    """
    Get the baselines from a header file.
    
    Extracts the baseline value from each signal definition line. If not
    explicitly specified, uses the ADC zero value.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of baseline values, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[int] = []
    adc_zeros = get_adc_zeros(header)
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 2:
            field = parts[2]
            # Extract baseline from 'gain/unit(baseline)' format
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                try:
                    baseline_str = field.split('(')[1].split(')')[0]
                    values.append(int(baseline_str))
                except (ValueError, IndexError):
                    values.append(adc_zeros[i - 1] if i - 1 < len(adc_zeros) else 0)
            else:
                values.append(adc_zeros[i - 1] if i - 1 < len(adc_zeros) else 0)
        else:
            values.append(adc_zeros[i - 1] if i - 1 < len(adc_zeros) else 0)
    
    return values


def get_signal_units(header: str) -> List[str]:
    """
    Get the signal units from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of signal unit strings (e.g., 'mV'), one per signal.
        Defaults to 'mV' if not specified.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[str] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 2:
            field = parts[2]
            if '/' in field:
                values.append(field.split('/')[1])
            else:
                values.append('mV')
        else:
            values.append('mV')
    
    return values


def get_adc_resolutions(header: str) -> List[int]:
    """
    Get the ADC resolutions from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of ADC resolution values (bits), one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[int] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 3:
            try:
                values.append(int(parts[3]))
            except (ValueError, IndexError):
                values.append(0)
        else:
            values.append(0)
    
    return values


def get_adc_zeros(header: str) -> List[int]:
    """
    Get the ADC zeros from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of ADC zero values, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[int] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 4:
            try:
                values.append(int(parts[4]))
            except (ValueError, IndexError):
                values.append(0)
        else:
            values.append(0)
    
    return values


def get_initial_values(header: str) -> List[int]:
    """
    Get the initial values of signals from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of initial signal values, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[int] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 5:
            try:
                values.append(int(parts[5]))
            except (ValueError, IndexError):
                values.append(0)
        else:
            values.append(0)
    
    return values


def get_checksums(header: str) -> List[int]:
    """
    Get the checksums of signals from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of checksum values, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[int] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 6:
            try:
                values.append(int(parts[6]))
            except (ValueError, IndexError):
                values.append(0)
        else:
            values.append(0)
    
    return values


def get_block_sizes(header: str) -> List[int]:
    """
    Get the block sizes of signals from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of block size values, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[int] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 7:
            try:
                values.append(int(parts[7]))
            except (ValueError, IndexError):
                values.append(0)
        else:
            values.append(0)
    
    return values


def get_signal_names(header: str) -> List[str]:
    """
    Get the signal names from a header file.
    
    Args:
        header: Header file content as a string.
        
    Returns:
        List of signal names, one per signal.
    """
    num_signals = get_num_signals(header)
    if num_signals is None:
        return []
    
    lines = header.split('\n')
    values: List[str] = []
    
    for i, line in enumerate(lines[1:num_signals + 1], start=1):
        parts = line.split()
        if len(parts) > 8:
            values.append(parts[8])
        else:
            values.append('')
    
    return values

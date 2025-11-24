"""
Data I/O operations for loading and saving records, signals, images, and labels.
"""

import os
import wfdb
from PIL import Image
from . import constants
from . import text_utils
from . import header_parser


# Find the records in a folder and its subfolders.
def find_records(folder):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                records.add(record)
    records = sorted(records)
    return records


# Load the header for a record.
def load_header(record):
    header_file = header_parser.get_header_file(record)
    header = text_utils.load_text(header_file)
    return header


# Load the signals for a record.
def load_signals(record):
    signal_files = header_parser.get_signal_files(record)
    path = os.path.split(record)[0]
    signal_files_exist = all(os.path.isfile(os.path.join(path, signal_file)) for signal_file in signal_files)
    if signal_files and signal_files_exist:
        signal, fields = wfdb.rdsamp(record)
    else:
        signal, fields = None, None
    return signal, fields


# Load the images for a record.
def load_images(record):
    path = os.path.split(record)[0]
    image_files = header_parser.get_image_files(record)

    images = list()
    for image_file in image_files:
        image_file_path = os.path.join(path, image_file)
        if os.path.isfile(image_file_path):
            image = Image.open(image_file_path)
            images.append(image)

    return images


# Load the labels for a record.
def load_labels(record):
    header = load_header(record)
    labels = header_parser.get_labels_from_header(header)
    return labels


# Save the header for a record.
def save_header(record, header):
    header_file = header_parser.get_header_file(record)
    text_utils.save_text(header_file, header)


# Save the signals for a record.
def save_signals(record, signal, comments=list()):
    header = load_header(record)
    path, record = os.path.split(record)
    sampling_frequency = header_parser.get_sampling_frequency(header)
    signal_formats = header_parser.get_signal_formats(header)
    adc_gains = header_parser.get_adc_gains(header)
    baselines = header_parser.get_baselines(header)
    signal_units = header_parser.get_signal_units(header)
    signal_names = header_parser.get_signal_names(header)
    comments = [comment.replace('#', '').strip() for comment in comments]

    wfdb.wrsamp(record, fs=sampling_frequency, units=signal_units, sig_name=signal_names, \
                p_signal=signal, fmt=signal_formats, adc_gain=adc_gains, baseline=baselines, comments=comments,
                write_dir=path)


# Save the labels for a record.
def save_labels(record, labels):
    header_file = header_parser.get_header_file(record)
    header = text_utils.load_text(header_file)
    header += constants.substring_labels + ' ' + ', '.join(labels) + '\n'
    text_utils.save_text(header_file, header)
    return header


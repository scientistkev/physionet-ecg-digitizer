"""
Header parsing functions for challenge-specific and WFDB header files.
"""

import os
from . import constants
from . import text_utils
from . import helpers


# Get the signal files from a header or a similar string.
def get_signal_files_from_header(string):
    signal_files = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i==0 and not l.startswith('#'):
            num_channels = int(arrs[1])
        elif i<=num_channels and not l.startswith('#'):
            signal_file = arrs[0]
            if signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
    return signal_files


# Get the image files from a header or a similar string.
def get_image_files_from_header(string):
    images, has_image = text_utils.get_variables(string, constants.substring_images)
    if not has_image:
        raise Exception('No images available: did you forget to generate or include the images?')
    return images


# Get the labels from a header or a similar string.
def get_labels_from_header(string):
    labels, has_labels = text_utils.get_variables(string, constants.substring_labels)
    if not has_labels:
        raise Exception('No labels available: are you trying to load the labels from the held-out data, or did you forget to prepare the data to include the labels?')
    return labels


# Get the header file for a record.
def get_header_file(record):
    if not record.endswith('.hea'):
        header_file = record + '.hea'
    else:
        header_file = record
    return header_file


# Get the signal files for a record.
def get_signal_files(record):
    header_file = get_header_file(record)
    header = text_utils.load_text(header_file)
    signal_files = get_signal_files_from_header(header)
    return signal_files


# Get the image files for a record.
def get_image_files(record):
    header_file = get_header_file(record)
    header = text_utils.load_text(header_file)
    image_files = get_image_files_from_header(header)
    return image_files


### WFDB functions

# Get the record name from a header file.
def get_record_name(string):
    value = string.split('\n')[0].split(' ')[0].split('/')[0].strip()
    return value


# Get the number of signals from a header file.
def get_num_signals(string):
    value = string.split('\n')[0].split(' ')[1].strip()
    if helpers.is_integer(value):
        value = int(value)
    else:
        value = None
    return value


# Get the sampling frequency from a header file.
def get_sampling_frequency(string):
    value = string.split('\n')[0].split(' ')[2].split('/')[0].strip()
    if helpers.is_number(value):
        value = float(value)
    else:
        value = None
    return value


# Get the number of samples from a header file.
def get_num_samples(string):
    value = string.split('\n')[0].split(' ')[3].strip()
    if helpers.is_integer(value):
        value = int(value)
    else:
        value = None
    return value


# Get the signal formats from a header file.
def get_signal_formats(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[1]
            if 'x' in field:
                field = field.split('x')[0]
            if ':' in field:
                field = field.split(':')[0]
            if '+' in field:
                field = field.split('+')[0]
            value = field
            values.append(value)
    return values


# Get the ADC gains from a header file.
def get_adc_gains(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[0]
            value = float(field)
            values.append(value)
    return values


# Get the baselines from a header file.
def get_baselines(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[1].split(')')[0]
            else:
                field = get_adc_zeros(string)[i-1]
            value = int(field)
            values.append(value)
    return values


# Get the signal units from a header file.
def get_signal_units(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                value = field.split('/')[1]
            else:
                value = 'mV'
            values.append(value)
    return values


# Get the ADC resolutions from a header file.
def get_adc_resolutions(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[3]
            value = int(field)
            values.append(value)
    return values


# Get the ADC zeros from a header file.
def get_adc_zeros(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[4]
            value = int(field)
            values.append(value)
    return values


# Get the initial values of a signal from a header file.
def get_initial_values(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[5]
            value = int(field)
            values.append(value)
    return values


# Get the checksums of a signal from a header file.
def get_checksums(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[6]
            value = int(field)
            values.append(value)
    return values


# Get the block sizes of a signal from a header file.
def get_block_sizes(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[7]
            value = int(field)
            values.append(value)
    return values


# Get the signal names from a header file.
def get_signal_names(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            value = l.split(' ')[8]
            values.append(value)
    return values


"""
Utils package for PhysioNet ECG digitization challenge.

This package provides helper functions organized into modules:
- constants: Challenge constants
- io: Data I/O operations
- text_utils: Text file utilities
- header_parser: Header parsing functions
- signal_processing: Signal processing functions
- evaluation: Evaluation metrics
- helpers: General utility functions
"""

# Constants
from . import constants
from .constants import substring_labels, substring_images

# Text utilities
from . import text_utils
from .text_utils import (
    load_text,
    save_text,
    get_variable,
    get_variables,
)

# Helper functions
from . import helpers
from .helpers import (
    is_number,
    is_integer,
    is_finite_number,
    is_nan,
    cast_int_float_unknown,
    compute_one_hot_encoding,
)

# Header parser
from . import header_parser
from .header_parser import (
    get_signal_files_from_header,
    get_image_files_from_header,
    get_labels_from_header,
    get_header_file,
    get_signal_files,
    get_image_files,
    get_record_name,
    get_num_signals,
    get_sampling_frequency,
    get_num_samples,
    get_signal_formats,
    get_adc_gains,
    get_baselines,
    get_signal_units,
    get_adc_resolutions,
    get_adc_zeros,
    get_initial_values,
    get_checksums,
    get_block_sizes,
    get_signal_names,
)

# Signal processing
from . import signal_processing
from .signal_processing import (
    normalize_names,
    reorder_signal,
    convert_signal,
    fft_correlate,
    align_signals,
)

# Evaluation
from . import evaluation
from .evaluation import (
    compute_one_vs_rest_confusion_matrix,
    compute_f_measure,
    compute_snr,
    compute_ks_metric,
    compute_asci_metric,
    compute_weighted_absolute_difference,
)

# I/O operations
from . import io
from .io import (
    find_records,
    load_header,
    load_signals,
    load_images,
    load_labels,
    save_header,
    save_signals,
    save_labels,
)

__all__ = [
    # Constants
    'substring_labels',
    'substring_images',
    # Text utilities
    'load_text',
    'save_text',
    'get_variable',
    'get_variables',
    # Helper functions
    'is_number',
    'is_integer',
    'is_finite_number',
    'is_nan',
    'cast_int_float_unknown',
    'compute_one_hot_encoding',
    # Header parser
    'get_signal_files_from_header',
    'get_image_files_from_header',
    'get_labels_from_header',
    'get_header_file',
    'get_signal_files',
    'get_image_files',
    'get_record_name',
    'get_num_signals',
    'get_sampling_frequency',
    'get_num_samples',
    'get_signal_formats',
    'get_adc_gains',
    'get_baselines',
    'get_signal_units',
    'get_adc_resolutions',
    'get_adc_zeros',
    'get_initial_values',
    'get_checksums',
    'get_block_sizes',
    'get_signal_names',
    # Signal processing
    'normalize_names',
    'reorder_signal',
    'convert_signal',
    'fft_correlate',
    'align_signals',
    # Evaluation
    'compute_one_vs_rest_confusion_matrix',
    'compute_f_measure',
    'compute_snr',
    'compute_ks_metric',
    'compute_asci_metric',
    'compute_weighted_absolute_difference',
    # I/O operations
    'find_records',
    'load_header',
    'load_signals',
    'load_images',
    'load_labels',
    'save_header',
    'save_signals',
    'save_labels',
]


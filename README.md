# physionet-ecg-digitizer

ECG image-to-signal extraction pipeline for the PhysioNet digitization challenge.

## Utils Package

The `utils` package provides helper functions organized into the following modules:

- **`constants.py`** - Challenge constants (label and image substring identifiers)
- **`helpers.py`** - General utility functions for type checking, casting, and one-hot encoding
- **`text_utils.py`** - Text file I/O operations and string parsing utilities
- **`header_parser.py`** - Functions for parsing WFDB header files and extracting metadata (signals, images, labels, signal properties)
- **`signal_processing.py`** - Signal processing functions including channel normalization, reordering, quantization, and temporal alignment
- **`evaluation.py`** - Evaluation metrics for signal quality assessment (SNR, KS metric, ASCI) and classification performance (F-measure, confusion matrices)
- **`io.py`** - Data I/O operations for loading and saving records, signals, images, and labels

All functions include comprehensive type hints and documentation. Import functions directly from the `utils` package:

```python
from utils import load_signals, compute_snr, align_signals
```

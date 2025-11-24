"""
Signal processing functions for normalization, reordering, quantization, and alignment.

This module provides functions for processing ECG signals including channel
normalization, signal reordering, amplitude quantization, and temporal alignment.
"""

from typing import List, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter


def normalize_names(names_ref: List[str], names_est: List[str]) -> List[str]:
    """
    Normalize channel names by matching estimated names to reference names
    using case-insensitive comparison.
    
    Args:
        names_ref: List of reference channel names.
        names_est: List of estimated channel names to normalize.
        
    Returns:
        List of normalized channel names from names_ref that match names_est.
    """
    normalized: List[str] = []
    ref_lower = {name.casefold(): name for name in names_ref}
    
    for est_name in names_est:
        est_lower = est_name.casefold()
        if est_lower in ref_lower:
            normalized.append(ref_lower[est_lower])
    
    return normalized


def reorder_signal(
    input_signal: Union[npt.NDArray, List[List[float]]],
    input_channels: List[str],
    output_channels: List[str]
) -> npt.NDArray:
    """
    Reorder channels in a signal array to match the desired channel order.
    
    Args:
        input_signal: Input signal array of shape (num_samples, num_channels).
        input_channels: List of channel names for the input signal.
        output_channels: List of desired channel names in output order.
        
    Returns:
        Reordered signal array of shape (num_samples, len(output_channels)).
        
    Raises:
        AssertionError: If input_channels or output_channels contain duplicates.
    """
    # Do not allow repeated channels with potentially different values in a signal.
    assert len(set(input_channels)) == len(input_channels), \
        "input_channels contains duplicates"
    assert len(set(output_channels)) == len(output_channels), \
        "output_channels contains duplicates"

    if input_channels == output_channels:
        return np.asarray(input_signal)

    # Normalize output channel names to match input
    normalized_output = normalize_names(input_channels, output_channels)
    
    input_signal_arr = np.asarray(input_signal)
    num_samples = input_signal_arr.shape[0]
    num_channels = len(normalized_output)
    data_type = input_signal_arr.dtype

    # Create mapping from output to input channel indices
    channel_map = {name: idx for idx, name in enumerate(input_channels)}
    
    output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
    for i, output_channel in enumerate(normalized_output):
        if output_channel in channel_map:
            input_idx = channel_map[output_channel]
            output_signal[:, i] = input_signal_arr[:, input_idx]

    return output_signal


def convert_signal(
    x: npt.NDArray[np.floating],
    num_quant_levels: int,
    min_amp: float,
    max_amp: float,
    max_t: int
) -> npt.NDArray[np.floating]:
    """
    Quantize 1D signal amplitudes to convert a real-valued signal to a 2D binarized signal.
    
    The signal is quantized into num_quant_levels bins between min_amp and max_amp,
    and represented as a 2D array where each column represents a time point and
    each row represents a quantization level.
    
    Args:
        x: 1D input signal array.
        num_quant_levels: Number of quantization levels.
        min_amp: Minimum amplitude value for quantization.
        max_amp: Maximum amplitude value for quantization.
        max_t: Maximum time index (number of columns in output).
        
    Returns:
        2D binary array of shape (num_quant_levels, max_t) where A[y-1, t-1] = 1
        indicates that at time t, the signal value falls in quantization level y.
    """
    x_arr = np.asarray(x)
    idx = np.isfinite(x_arr)

    t = np.arange(1, x_arr.size + 1)
    t = t[idx]

    y = x_arr[idx]
    # Quantize: map [min_amp, max_amp] to [1, num_quant_levels]
    y = np.round((num_quant_levels - 1) * (y - min_amp) / (max_amp - min_amp) + 1).astype(int)
    y = np.clip(y, 1, num_quant_levels)

    A = np.zeros((num_quant_levels, max_t), dtype=np.float64)
    A[y - 1, t - 1] = 1.0
    return A


def fft_correlate(
    A_ref: npt.NDArray[np.floating],
    A_est: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Correlate 2D signals in the spectral domain using FFT-based convolution.
    
    Args:
        A_ref: 2D reference signal array.
        A_est: 2D estimated signal array.
        
    Returns:
        2D correlation array computed via FFT convolution.
    """
    # Flip the digitized signal for correlation
    A_est_flipped = np.flip(np.flip(A_est, axis=0), axis=1)
    return fftconvolve(A_ref, A_est_flipped, mode='full')


def align_signals(
    x_ref: npt.NDArray[np.floating],
    x_est: npt.NDArray[np.floating],
    num_quant_levels: int,
    smooth: bool = True,
    sigma: float = 0.5
) -> Tuple[npt.NDArray[np.floating], int, float]:
    """
    Estimate vertical and horizontal offsets of the estimated signal vs. a reference signal.
    
    Uses 2D correlation in the spectral domain to find optimal alignment between
    reference and estimated signals. The method quantizes both signals, optionally
    applies Gaussian smoothing, and finds the correlation peak to determine offsets.
    
    Reference: Reza Sameni, Zuzana Koscova, Matthew Reyna, July 2024
    
    Args:
        x_ref: 1D reference signal array.
        x_est: 1D estimated signal array.
        num_quant_levels: Number of quantization levels for 2D representation.
        smooth: Whether to apply Gaussian smoothing to quantized signals. Default True.
        sigma: Standard deviation for Gaussian smoothing. Default 0.5.
        
    Returns:
        Tuple of (x_est_shifted, offset_hz, offset_vt) where:
        - x_est_shifted: Estimated signal shifted by the computed offsets
        - offset_hz: Horizontal (temporal) offset in samples
        - offset_vt: Vertical (amplitude) offset
    """
    x_ref_arr = np.asarray(x_ref)
    x_est_arr = np.asarray(x_est)
    
    # Summarize the durations and amplitudes of the signals
    min_amp = min(np.nanmin(x_ref_arr), np.nanmin(x_est_arr))
    max_amp = max(np.nanmax(x_ref_arr), np.nanmax(x_est_arr))
    max_t = max(x_ref_arr.size, x_est_arr.size)

    # Quantize the 1D signal amplitudes to convert to 2D binarized signals
    A_ref = convert_signal(x_ref_arr, num_quant_levels, min_amp, max_amp, max_t)
    A_est = convert_signal(x_est_arr, num_quant_levels, min_amp, max_amp, max_t)

    # Apply Gaussian smoothing to the 2D binarized signals (optional)
    if smooth:
        A_ref = gaussian_filter(A_ref, sigma)
        A_est = gaussian_filter(A_est, sigma)
    
    # Compute the cross-correlation of 2D reference and estimated signals
    A_cross = fft_correlate(A_ref, A_est)
    idx_cross = np.unravel_index(np.argmax(A_cross), A_cross.shape)
                                 
    # Compute the auto-correlation of the reference signal
    A_auto = fft_correlate(A_ref, A_ref)
    idx_auto = np.unravel_index(np.argmax(A_auto), A_auto.shape)
   
    # Estimate vertical and horizontal offsets from the cross-correlation peak lags
    offset_hz = int(idx_auto[1] - idx_cross[1])
    offset_vt = idx_auto[0] - idx_cross[0]
    offset_vt = offset_vt / (num_quant_levels - 1) * (max_amp - min_amp)

    # Shift the estimated signal by the estimated offsets
    if offset_hz < 0:
        x_est_shifted = np.concatenate((np.nan * np.ones(-offset_hz), x_est_arr))
    else:
        x_est_shifted = np.concatenate((x_est_arr[offset_hz:], np.nan * np.ones(offset_hz)))
    x_est_shifted = x_est_shifted - offset_vt

    return x_est_shifted, offset_hz, float(offset_vt)

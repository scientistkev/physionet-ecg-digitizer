"""
Signal processing functions for normalization, reordering, quantization, and alignment.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter


# Normalize the channel names.
def normalize_names(names_ref, names_est):
    tmp = list()
    for a in names_est:
        for b in names_ref:
            if a.casefold() == b.casefold():
                tmp.append(b)
                break
    return tmp


# Reorder channels in signal.
def reorder_signal(input_signal, input_channels, output_channels):
    # Do not allow repeated channels with potentially different values in a signal.
    assert(len(set(input_channels)) == len(input_channels))
    assert(len(set(output_channels)) == len(output_channels))

    if input_channels == output_channels:
        output_signal = input_signal
    else:
        output_channels = normalize_names(input_channels, output_channels)

        input_signal = np.asarray(input_signal)
        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype

        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] = input_signal[:, j]

    return output_signal


# Quantize the 1D signal amplitudes to convert the 1D real-valued signal to a 2D binarized signal.
def convert_signal(x, num_quant_levels, min_amp, max_amp, max_t):
    idx = np.isfinite(x)

    t = np.arange(1, np.size(x) + 1)
    t = t[idx]

    y = x[idx]
    y = np.round((num_quant_levels - 1) * (y - min_amp) / (max_amp - min_amp) + 1).astype(int)
    y = np.clip(y, 1, num_quant_levels)

    A = np.zeros((num_quant_levels, max_t))
    A[y - 1, t - 1] = 1
    return A


# Correlate the 2D signals in the spectral domain.
def fft_correlate(A_ref, A_est):
    # Flip the digitized signal for correlation. 
    A_est_flipped = np.flip(np.flip(A_est, axis=0), axis=1)
    return fftconvolve(A_ref, A_est_flipped, mode='full')


def align_signals(x_ref, x_est, num_quant_levels, smooth=True, sigma=0.5):
    # Estimate the vertical and horizontal offsets of the estimated signal vs. a reference signal
    # in noisy conditions.
    # Reza Sameni, Zuzana Koscova, Matthew Reyna, July 2024

    # Summarize the durations and amplitudes of the signals.
    min_amp = min(np.nanmin(x_ref), np.nanmin(x_est))
    max_amp = max(np.nanmax(x_ref), np.nanmax(x_est))
    max_t = max(np.size(x_ref), np.size(x_est))

    # Quantize the 1D signal amplitudes to convert the 1D real-valued signals to 2D binarized signals.
    A_ref = convert_signal(x_ref, num_quant_levels, min_amp, max_amp, max_t)
    A_est = convert_signal(x_est, num_quant_levels, min_amp, max_amp, max_t)

    # Apply Gaussian smoothing to the 2D binarized signals (optional).
    if smooth:
        A_ref = gaussian_filter(A_ref, sigma)
        A_est = gaussian_filter(A_est, sigma)
    
    # Compute the cross-correlation of 2D reference and estimated signals in the spectral domain.
    A_cross = fft_correlate(A_ref, A_est)
    idx_cross = np.unravel_index(np.argmax(A_cross), A_cross.shape)
                                 
    # Compute the auto-correlation of the reference signal in the spectral domain.
    A_auto = fft_correlate(A_ref, A_ref)
    idx_auto = np.unravel_index(np.argmax(A_auto), A_auto.shape)
   
    # Estimate vertical and horizontal offsets from the cross-correlation peak lags.
    offset_hz = idx_auto[1] - idx_cross[1]
    offset_vt = idx_auto[0] - idx_cross[0]
    offset_vt = offset_vt / (num_quant_levels - 1) * (max_amp - min_amp)

    # Shift the estimated signal by the estimated offsets.
    if offset_hz < 0:
        x_est_shifted = np.concatenate((np.nan*np.ones(-offset_hz), x_est))
    else:
        x_est_shifted = np.concatenate((x_est[offset_hz:], np.nan*np.ones(offset_hz)))
    x_est_shifted -= offset_vt

    return x_est_shifted, offset_hz, offset_vt


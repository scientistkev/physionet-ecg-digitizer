"""
Evaluation metrics for signal quality assessment and classification performance.

This module provides functions for computing various evaluation metrics including
confusion matrices, F-measure, signal-to-noise ratio, and other signal quality metrics.
"""

from typing import Tuple, List, Sequence, Any, Union
import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt
from . import helpers


def compute_one_vs_rest_confusion_matrix(
    labels: npt.NDArray[np.integer],
    outputs: npt.NDArray[np.integer],
    classes: Sequence[Any]
) -> npt.NDArray[np.floating]:
    """
    Construct binary one-vs-rest confusion matrices for each class.
    
    For each class, creates a 2x2 confusion matrix where:
    - Columns represent expert labels (ground truth)
    - Rows represent classifier outputs (predictions)
    
    Args:
        labels: Binary array of shape (num_instances, num_classes) with ground truth labels.
        outputs: Binary array of shape (num_instances, num_classes) with predictions.
        classes: Sequence of class labels (used to determine num_classes).
        
    Returns:
        3D array of shape (num_classes, 2, 2) where A[k, :, :] is the confusion
        matrix for class k. The matrix structure is:
        [[TP, FP],
         [FN, TN]]
        
    Raises:
        AssertionError: If labels and outputs have different shapes.
    """
    assert labels.shape == outputs.shape, \
        f"Labels shape {labels.shape} != outputs shape {outputs.shape}"

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2), dtype=np.float64)
    
    # Vectorized computation for better efficiency
    for j in range(num_classes):
        label_col = labels[:, j]
        output_col = outputs[:, j]
        
        # True Positives: both label and output are 1
        A[j, 0, 0] = np.sum((label_col == 1) & (output_col == 1))
        # False Positives: label is 0 but output is 1
        A[j, 0, 1] = np.sum((label_col == 0) & (output_col == 1))
        # False Negatives: label is 1 but output is 0
        A[j, 1, 0] = np.sum((label_col == 1) & (output_col == 0))
        # True Negatives: both label and output are 0
        A[j, 1, 1] = np.sum((label_col == 0) & (output_col == 0))

    return A


def compute_f_measure(
    labels: Sequence[Sequence[Any]],
    outputs: Sequence[Sequence[Any]]
) -> Tuple[float, npt.NDArray[np.floating], List[Any]]:
    """
    Compute macro F-measure (F1 score) for multi-label classification.
    
    The F-measure is computed for each class individually, then averaged
    to get the macro F-measure.
    
    Args:
        labels: Sequence of sequences, where each inner sequence contains
            ground truth labels for one instance.
        outputs: Sequence of sequences, where each inner sequence contains
            predicted labels for one instance.
            
    Returns:
        Tuple of (macro_f_measure, per_class_f_measure, classes) where:
        - macro_f_measure: Average F-measure across all classes
        - per_class_f_measure: Array of F-measures for each class
        - classes: List of all unique class labels found in labels
    """
    # Compute confusion matrix
    classes = sorted(set.union(*map(set, labels)))
    labels_encoded = helpers.compute_one_hot_encoding(labels, classes)
    outputs_encoded = helpers.compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels_encoded, outputs_encoded, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes, dtype=np.float64)
    
    for k in range(num_classes):
        tp, fp, fn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0]
        denominator = 2 * tp + fp + fn
        if denominator > 0:
            per_class_f_measure[k] = (2 * tp) / denominator
        else:
            per_class_f_measure[k] = np.nan

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = float(np.nanmean(per_class_f_measure))
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, classes


def _align_signals_for_comparison(
    x_ref: npt.NDArray[np.floating],
    x_est: npt.NDArray[np.floating],
    keep_nans: bool
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """
    Align two signals for comparison by padding with NaNs and handling non-finite values.
    
    Args:
        x_ref: Reference signal array.
        x_est: Estimated signal array.
        keep_nans: If True, only compare samples with finite values in both signals.
                   If False, replace non-finite values in x_est with zeros.
                   
    Returns:
        Tuple of (x_ref_aligned, x_est_aligned, valid_idx) where valid_idx
        indicates which samples should be used for comparison.
    """
    x_ref_arr = np.asarray(x_ref, dtype=np.float64).copy()
    x_est_arr = np.asarray(x_est, dtype=np.float64).copy()
    
    assert x_ref_arr.ndim == x_est_arr.ndim == 1, \
        "Signals must be 1-dimensional"

    # Pad the shorter signal with NaNs
    n_ref = x_ref_arr.size
    n_est = x_est_arr.size
    if n_est < n_ref:
        x_est_arr = np.concatenate((x_est_arr, np.nan * np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref_arr = np.concatenate((x_ref_arr, np.nan * np.ones(n_est - n_ref)))

    # Identify samples with finite values
    idx_ref = np.isfinite(x_ref_arr)
    idx_est = np.isfinite(x_est_arr)

    # Determine valid comparison indices
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est_arr[~idx_est] = 0.0
        idx = idx_ref

    return x_ref_arr[idx], x_est_arr[idx], idx


def compute_snr(
    x_ref: npt.NDArray[np.floating],
    x_est: npt.NDArray[np.floating],
    keep_nans: bool = True,
    signal_median: bool = False,
    noise_median: bool = False
) -> Tuple[float, float, float]:
    """
    Compute signal-to-noise ratio (SNR) between reference and estimated signals.
    
    SNR is computed as 10 * log10(p_signal / p_noise), where p_signal and p_noise
    are the power of the signal and noise respectively.
    
    Args:
        x_ref: Reference signal array.
        x_est: Estimated signal array.
        keep_nans: If True, only consider samples with finite values in both signals
                   and penalize missing samples. If False, replace non-finite values
                   in x_est with zeros. Default True.
        signal_median: If True, use median for signal power calculation.
                      If False, use mean. Default False.
        noise_median: If True, use median for noise power calculation.
                     If False, use mean. Default False.
        
    Returns:
        Tuple of (snr, p_signal, p_noise) where:
        - snr: Signal-to-noise ratio in dB
        - p_signal: Signal power
        - p_noise: Noise power
    """
    x_ref_aligned, x_est_aligned, idx = _align_signals_for_comparison(
        x_ref, x_est, keep_nans
    )

    # Compute the noise
    x_noise = x_ref_aligned - x_est_aligned

    # Compute power using mean or median
    if signal_median:
        p_signal = float(np.median(x_ref_aligned ** 2))
    else:
        p_signal = float(np.mean(x_ref_aligned ** 2))

    if noise_median:
        p_noise = float(np.median(x_noise ** 2))
    else:
        p_noise = float(np.mean(x_noise ** 2))

    # Compute SNR
    if p_signal > 0 and p_noise > 0:
        snr = 10 * np.log10(p_signal / p_noise)
    else:
        snr = float('nan')

    # Penalize missing samples if keep_nans is True
    if keep_nans:
        x_ref_full = np.asarray(x_ref, dtype=np.float64)
        idx_ref = np.isfinite(x_ref_full)
        alpha = np.sum(idx) / np.sum(idx_ref) if np.sum(idx_ref) > 0 else 0.0
        snr *= alpha

    return float(snr), p_signal, p_noise


def compute_ks_metric(
    x_ref: npt.NDArray[np.floating],
    x_est: npt.NDArray[np.floating],
    keep_nans: bool = True
) -> float:
    """
    Compute a metric inspired by the Kolmogorov-Smirnov test statistic.
    
    The metric compares the cumulative distribution functions (CDFs) of the
    absolute values of the reference and estimated signals. Returns a
    goodness-of-fit measure between 0 and 1.
    
    Args:
        x_ref: Reference signal array.
        x_est: Estimated signal array.
        keep_nans: If True, only consider samples with finite values in both signals.
                   If False, replace non-finite values in x_est with zeros.
                   Default True.
        
    Returns:
        Goodness-of-fit metric between 0 and 1, where 1 indicates perfect match.
    """
    x_ref_aligned, x_est_aligned, _ = _align_signals_for_comparison(
        x_ref, x_est, keep_nans
    )

    # Compute cumulative sums (CDFs) of absolute values
    x_ref_cdf = np.nancumsum(np.abs(x_ref_aligned))
    x_est_cdf = np.nancumsum(np.abs(x_est_aligned))

    # Normalize CDFs
    if x_ref_cdf[-1] > 0:
        x_ref_cdf = x_ref_cdf / x_ref_cdf[-1]
    if x_est_cdf[-1] > 0:
        x_est_cdf = x_est_cdf / x_est_cdf[-1]

    # Compute maximum difference (KS statistic)
    goodness_of_fit = 1.0 - float(np.max(np.abs(x_ref_cdf - x_est_cdf)))

    return goodness_of_fit


def compute_asci_metric(
    x_ref: npt.NDArray[np.floating],
    x_est: npt.NDArray[np.floating],
    beta: float = 0.05,
    keep_nans: bool = True
) -> float:
    """
    Compute the adaptive signed correlation index (ASCI) metric.
    
    ASCI discretizes the noise based on a threshold (beta * std(x_ref)) and
    computes the mean of the discretized values. Values within threshold get +1,
    values above threshold get -1.
    
    Args:
        x_ref: Reference signal array.
        x_est: Estimated signal array.
        beta: Threshold parameter (0 < beta <= 1). Default 0.05.
        keep_nans: If True, only consider samples with finite values in both signals.
                   If False, replace non-finite values in x_est with zeros.
                   Default True.
        
    Returns:
        ASCI metric value between -1 and 1, where higher values indicate better match.
        
    Raises:
        ValueError: If beta is not in (0, 1].
    """
    if beta <= 0 or beta > 1:
        raise ValueError('The beta value should be greater than 0 and less than or equal to 1.')

    x_ref_aligned, x_est_aligned, _ = _align_signals_for_comparison(
        x_ref, x_est, keep_nans
    )

    # Compute threshold and discretize noise
    threshold = beta * float(np.std(x_ref_aligned))
    x_noise = np.abs(x_ref_aligned - x_est_aligned)

    # Discretize: 1 if noise <= threshold, -1 otherwise
    x_noise_discretized = np.where(x_noise <= threshold, 1.0, -1.0)

    asci = float(np.mean(x_noise_discretized))
    return asci


def compute_weighted_absolute_difference(
    x_ref: npt.NDArray[np.floating],
    x_est: npt.NDArray[np.floating],
    sampling_frequency: float,
    keep_nans: bool = True
) -> float:
    """
    Compute a weighted absolute difference metric between signals.
    
    The metric applies a low-pass filter to the reference signal to create
    weights, then computes a weighted mean absolute difference.
    
    Args:
        x_ref: Reference signal array.
        x_est: Estimated signal array.
        sampling_frequency: Sampling frequency in Hz.
        keep_nans: If True, only consider samples with finite values in both signals.
                   If False, replace non-finite values in x_est with zeros.
                   Default True.
        
    Returns:
        Weighted absolute difference metric (lower is better).
    """
    x_ref_aligned, x_est_aligned, _ = _align_signals_for_comparison(
        x_ref, x_est, keep_nans
    )

    # Filter the reference signal to compute weights
    m = round(0.1 * sampling_frequency)
    if m < 1:
        m = 1
    
    # Apply low-pass filter
    w = filtfilt(np.ones(m), m, x_ref_aligned, method='gust')
    w_max = np.max(w)
    if w_max > 0:
        w = 1 - 0.5 / w_max * w
    else:
        w = np.ones_like(w)
    
    n = np.sum(w)
    if n > 0:
        weighted_absolute_difference_metric = float(np.sum(np.abs(x_ref_aligned - x_est_aligned) * w) / n)
    else:
        weighted_absolute_difference_metric = float('nan')

    return weighted_absolute_difference_metric

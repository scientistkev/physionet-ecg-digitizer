"""
Evaluation metrics for signal quality assessment and classification performance.
"""

import numpy as np
from scipy.signal import filtfilt
from . import helpers


# Construct the binary one-vs-rest confusion matrices, where the columns are the expert labels and the rows are the classifier
# for the given classes.
def compute_one_vs_rest_confusion_matrix(labels, outputs, classes):
    assert np.shape(labels) == np.shape(outputs)

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_instances):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1: # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1: # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0: # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0: # TN
                A[j, 1, 1] += 1

    return A


# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    # Compute confusion matrix.
    classes = sorted(set.union(*map(set, labels)))
    labels = helpers.compute_one_hot_encoding(labels, classes)
    outputs = helpers.compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float('nan')

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, classes


def compute_snr(x_ref, x_est, keep_nans=True, signal_median=False, noise_median=False):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

    # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    # Compute the noise.
    x_noise = x_ref - x_est

    # Compute the power for the signal and the noise using either the mean (default) or the median.
    if not signal_median:
        p_signal = np.mean(x_ref**2)
    else:
        p_signal = np.median(x_ref**2)

    if not noise_median:
        p_noise = np.mean(x_noise**2)
    else:
        p_noise = np.median(x_noise**2)

    # Compute the SNR.
    if p_signal > 0 and p_noise > 0:
        snr = 10 * np.log10(p_signal / p_noise)
    else:
        snr = float('nan')

    # If only considering the samples with finite values in both signals, then penalize the samples with non-finite values in the
    # estimated signal but not in the reference signal.
    if keep_nans:
        alpha = np.sum(idx) / np.sum(idx_ref)
        snr *= alpha

    return snr, p_signal, p_noise


# Compute a metric inspired by the Kolmogorov-Smirnov test statistic.
def compute_ks_metric(x_ref, x_est, keep_nans=True):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

     # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    x_ref_cdf = np.nancumsum(np.abs(x_ref))
    x_est_cdf = np.nancumsum(np.abs(x_est))

    if x_ref_cdf[-1] > 0:
        x_ref_cdf = x_ref_cdf / x_ref_cdf[-1]
    if x_est_cdf[-1] > 0:
        x_est_cdf = x_est_cdf / x_est_cdf[-1]

    goodness_of_fit = 1.0 - np.max(np.abs(x_ref_cdf - x_est_cdf))

    return goodness_of_fit


# Compute the adaptive signed correlation index (ASCI) metric.
def compute_asci_metric(x_ref, x_est, beta=0.05, keep_nans=True):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

    # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    # Check the threshold parameter beta and discretize the nose.
    if beta <= 0 or beta > 1:
        raise ValueError('The beta value should be greater than 0 and less than or equal to 1.')

    threshold = beta * np.std(x_ref)

    x_noise = np.abs(x_ref - x_est)

    x_noise_discretized = np.zeros_like(x_noise)
    x_noise_discretized[x_noise <= threshold] = 1
    x_noise_discretized[x_noise > threshold] = -1

    asci = np.mean(x_noise_discretized)

    return asci


# Compute a weighted absolute difference metric.
def compute_weighted_absolute_difference(x_ref, x_est, sampling_frequency, keep_nans=True):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

    # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    # Filter the reference signal and compute a weighted absolute difference metric between the signals.
    m = round(0.1 * sampling_frequency)
    w = filtfilt(np.ones(m), m, x_ref, method='gust')
    w = 1 - 0.5/np.max(w) * w
    n = np.sum(w)

    weighted_absolute_difference_metric = np.sum(np.abs(x_ref - x_est) * w)/n

    return weighted_absolute_difference_metric


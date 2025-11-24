"""
General utility functions for type checking, casting, and encoding.
"""

from typing import Any, Union, List, Sequence
import numpy as np
import numpy.typing as npt


def is_number(x: Any) -> bool:
    """
    Check if a variable is a number or represents a number.
    
    Args:
        x: Variable to check.
        
    Returns:
        True if x can be converted to a float, False otherwise.
    """
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def is_integer(x: Any) -> bool:
    """
    Check if a variable is an integer or represents an integer.
    
    Args:
        x: Variable to check.
        
    Returns:
        True if x is an integer, False otherwise.
    """
    if is_number(x):
        return float(x).is_integer()
    return False


def is_finite_number(x: Any) -> bool:
    """
    Check if a variable is a finite number or represents a finite number.
    
    Args:
        x: Variable to check.
        
    Returns:
        True if x is a finite number, False otherwise.
    """
    if is_number(x):
        return np.isfinite(float(x))
    return False


def is_nan(x: Any) -> bool:
    """
    Check if a variable is a NaN (not a number) or represents a NaN.
    
    Args:
        x: Variable to check.
        
    Returns:
        True if x is NaN, False otherwise.
    """
    if is_number(x):
        return np.isnan(float(x))
    return False


def cast_int_float_unknown(x: Any) -> Union[int, float, str]:
    """
    Cast a value to an integer if an integer, a float if a non-integer float,
    and 'Unknown' if it's a number but not finite, otherwise raise an error.
    
    Args:
        x: Value to cast.
        
    Returns:
        Cast value as int, float, or 'Unknown'.
        
    Raises:
        NotImplementedError: If x cannot be cast to any of the supported types.
    """
    if is_integer(x):
        return int(x)
    if is_finite_number(x):
        return float(x)
    if is_number(x):
        return 'Unknown'
    raise NotImplementedError(f'Unable to cast {x}.')


def compute_one_hot_encoding(
    data: Sequence[Sequence[Any]], 
    classes: Sequence[Any]
) -> npt.NDArray[np.bool_]:
    """
    Construct the one-hot encoding of data for the given classes.
    
    Each element in data is a sequence of labels. The function creates a binary
    matrix where each row corresponds to an instance and each column to a class.
    A value of 1 indicates the instance has that class label.
    
    Args:
        data: Sequence of sequences, where each inner sequence contains labels
            for one instance.
        classes: Sequence of all possible class labels.
            
    Returns:
        Binary array of shape (num_instances, num_classes) with dtype bool.
    """
    num_instances = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_instances, num_classes), dtype=np.bool_)
    for i, instance_labels in enumerate(data):
        for label in instance_labels:
            for j, class_label in enumerate(classes):
                if (label == class_label) or (is_nan(label) and is_nan(class_label)):
                    one_hot_encoding[i, j] = True

    return one_hot_encoding


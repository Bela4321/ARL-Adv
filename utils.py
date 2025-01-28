import tensorflow as tf
import numpy as np

def inner_dot(a, b, axis=-1, keepdims=False, name=None):
    """
    Computes the sum of elementwise multiplication of a and b along the given axis.
    """
    with tf.name_scope(name or "inner_dot"):
        return tf.reduce_sum(a * b, axis=axis, keepdims=keepdims)

def l2_norm_tensors(*tensors):
    """
    L2-normalizes each tensor in `tensors` along the last dimension.
    Returns the same number of tensors.
    """
    return apply_tensors(lambda t: tf.nn.l2_normalize(t, axis=-1), *tensors)

def apply_tensors(func, *tensors):
    """
    Applies a given function `func` to each tensor in `tensors`.
    Returns a list if multiple tensors are provided, otherwise returns a single value.
    """
    res = [func(t) for t in tensors]
    return res if len(res) > 1 else res[0]

def pad_batch(list_of_seqs, dtype='int32'):
    """
    Given a list of integer lists (variable-length), pad them to the same max length.
    Returns a 2D NumPy array of shape [batch_size, max_length].
    """
    max_len = max(len(seq) for seq in list_of_seqs) if list_of_seqs else 0
    batch_size = len(list_of_seqs)
    arr = np.zeros((batch_size, max_len), dtype=dtype)
    for i, seq in enumerate(list_of_seqs):
        arr[i, :len(seq)] = seq
    return arr

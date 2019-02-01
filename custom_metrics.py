from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.metrics_impl import mean
from tensorflow.python.ops.metrics_impl import _remove_squeezable_dimensions


def word_level_accuracy(labels,
             predictions,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
    """Calculates how often `predictions` matches `labels`.
    The `accuracy` function creates two local variables, `total` and
    `count` that are used to compute the frequency with which `predictions`
    matches `labels`. This frequency is ultimately returned as `accuracy`: an
    idempotent operation that simply divides `total` by `count`.
    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the `accuracy`.
    Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
    where the corresponding elements of `predictions` and `labels` match and 0.0
    otherwise. Then `update_op` increments `total` with the reduced sum of the
    product of `weights` and `is_correct`, and it increments `count` with the
    reduced sum of `weights`.
    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
    Args:
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    predictions: The predicted values, a `Tensor` of any shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.
    Returns:
    accuracy: A `Tensor` representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.
    Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.accuracy is not supported when eager '
                           'execution is enabled.')

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=predictions, labels=labels, weights=weights
    )
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    is_correct = math_ops.reduce_prod(is_correct, axis=-1)
    return mean(is_correct, weights, metrics_collections, updates_collections, name or 'accuracy')


def top_5_accuracy(labels,
             predictions,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
    """Calculates how often `predictions` matches `labels`.
    The `accuracy` function creates two local variables, `total` and
    `count` that are used to compute the frequency with which `predictions`
    matches `labels`. This frequency is ultimately returned as `accuracy`: an
    idempotent operation that simply divides `total` by `count`.
    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the `accuracy`.
    Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
    where the corresponding elements of `predictions` and `labels` match and 0.0
    otherwise. Then `update_op` increments `total` with the reduced sum of the
    product of `weights` and `is_correct`, and it increments `count` with the
    reduced sum of `weights`.
    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
    Args:
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    predictions: The predicted values, a `Tensor` of any shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.
    Returns:
    accuracy: A `Tensor` representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.
    Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.accuracy is not supported when eager '
                           'execution is enabled.')

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=predictions, labels=labels, weights=weights
    )
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    is_correct = math_ops.reduce_prod(is_correct, axis=-2)
    is_correct = math_ops.reduce_max(is_correct, axis=-1)
    return mean(is_correct, weights, metrics_collections, updates_collections, name or 'accuracy')
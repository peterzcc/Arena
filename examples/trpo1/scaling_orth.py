from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.ops.init_ops import Initializer, _assert_float_dtype, _compute_fans


class ScalingOrth(Initializer):
    """Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is initialized
    with an orthogonal matrix obtained from the QR decomposition of a matrix of
    uniform random numbers. If the matrix has fewer rows than columns then the
    output will have orthogonal rows. Otherwise, the output will have orthogonal
    columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Args:
      gain: multiplicative factor to apply to the orthogonal matrix
      dtype: The type of the output.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
    """

    def __init__(self, scale=1.0, seed=None, dtype=dtypes.float32, mode="fan_avg", ):
        self.gain = scale
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
        self.seed = seed
        self.mode = mode

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        # Check the shape
        if len(shape) < 2:
            raise ValueError("The tensor to initialize must be "
                             "at least two-dimensional")
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows,
                                                                       num_cols)

        # Generate a random matrix
        a = random_ops.random_normal(flat_shape, dtype=dtype, seed=self.seed)
        # Compute the qr factorization
        q, r = linalg_ops.qr(a, full_matrices=False)
        # Make Q uniform
        d = array_ops.diag_part(r)
        ph = d / math_ops.abs(d)
        q *= ph
        if num_rows < num_cols:
            q = array_ops.matrix_transpose(q)

        scale = self.gain
        scale_shape = shape
        if partition_info is not None:
            scale_shape = partition_info.full_shape
        fan_in, fan_out = _compute_fans(scale_shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        return scale * array_ops.reshape(q, shape)

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name, "mode": self.mode}

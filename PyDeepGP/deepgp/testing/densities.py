# Copyright 2016 James Hensman, alexggmatthews
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import numpy as np

def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covariance.

    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(x) == 1 else tf.shape(x)[1]
    num_col = tf.cast(num_col, tf.float32)
    num_dims = tf.cast(tf.shape(x)[0], tf.float32)
    ret = - 0.5 * num_dims * num_col * np.log(2 * np.pi)
    ret += - num_col * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))
    return ret

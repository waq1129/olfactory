from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution
import matrixnormal_tf

try:
  from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class distributions_MatrixNormal(Distribution):
  """Matrix Variate random variable.
  #### Examples
  ```python
  # 100 samples of a scalar
  x = MVN(loc=tf.zeros(, rowcov, colcov)
  dp = MVN



  x = Empirical(params=tf.zeros(100))
  assert x.shape == ()
  # 5 samples of a 2 x 3 matrix
  dp = Empirical(params=tf.zeros([5, 2, 3]))
  assert x.shape == (2, 3)
  ```
  """
  def __init__(self,
               loc=None,
               rowcov=None,
               colcov=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MatrixNormal"):
    """Initialize a `Matrix Variate` random variable.
    Args:
      loc: 

      rowcov:

      colcov:




      params: tf.Tensor.
      Collection of samples. Its outer (left-most) dimension
      determines the number of samples.

    ****how to keep track of batches covariance matrices


    """
    parameters = locals()

    self._loc = tf.convert_to_tensor(loc, name='loc')
    self._rowcov = tf.convert_to_tensor(rowcov, name='rowcov')
    self._colcov = tf.convert_to_tensor(colcov, name='colcov')
    self._cov = tf.convert_to_tensor(rowcov, name='cov')
    self._n = tf.shape(self.loc)[0]

    #parameters = {'loc': self.loc, 'rowcov': self.rowcov, 'colcov': self.colcov}


    '''
    with tf.name_scope(name, values=[params]):
      with tf.control_dependencies([]):
        self._params = tf.identity(params, name="params")
        try:
          self._n = tf.shape(self._params)[0]
        except ValueError:  # scalar params
          self._n = tf.constant(1)
    '''

    super(distributions_MatrixNormal, self).__init__(
        dtype=self.loc.dtype,
        reparameterization_type=FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self.loc, self.rowcov, \
          self.colcov, self.n],
        name=name)

  @property
  def n(self):
    """Number of samples."""
    return self._n

  @property
  def loc(self):
    return self._loc
  
  @property
  def rowcov(self):
    return self._rowcov
  
  @property
  def colcov(self):
    return self._colcov
  
  def _mean(self):
    return self._loc
  
  def _covariance(self):
    return self._cov

  def _rowcov(self):
    return self._rowcov

  def _colcov(self):
    return self._colcov

  def _batch_shape_tensor(self):
    return array_ops.shape(self.loc)
    #return tf.constant([], dtype=tf.int32)

  def _batch_shape(self):
    return tf.TensorShape(self.loc.shape[:-1])
    #return tf.TensorShape([])

  def _event_shape_tensor(self):
    #return constant_op.constant([], dtype=dtypes.int32)
    return tf.shape(self.loc)[-1]

  def _event_shape(self):
    return tf.TensorShape(self.loc.shape[-1])
    #return tensor_shape.scalar()
    #return self.loc.shape[1:]
  

  def _log_prob(self, value):
    '''
      value,rowcov,colcov are of type numpy
    '''
    #import helpers
    #import matrixnormal
    #import covs
    import matrixnormal_tf

    val = matrixnormal_tf.matnorm_logp(value, self.loc, self.rowcov, self.colcov)
    return val


    '''
    def np_logpdf(value, loc, rowcov, colcov):
      import scipy.stats
      from scipy.stats import matrix_normal
      import numpy as np
      return matrix_normal.logpdf(X = value, mean = loc, rowcov = rowcov, colcov = colcov).astype(np.float32)

    # tf wrapper of python
    val = tf.py_func(np_logpdf, [value, self.loc, self.rowcov, self.colcov], [tf.float32])[0]
    #import pdb; pdb.set_trace()
    return tf.convert_to_tensor(tf.reshape(val, (1,)))[0]

    #rowcov_obj = covs.CovFullRankCholesky(size=self.reg_n, Sigma=rowcov)
    #colcov_obj = covs.CovFullRankCholesky(size=self.reg_n, Sigma=colcov)
    #return matrixnormal.matnorm_logp(value, rowcov_obj, colcov_obj)
    '''

  def _sample_n(self, nn, seed=None):
    def np_sample(loc, rowcov, colcov, nn):
      import scipy.stats
      from scipy.stats import matrix_normal
      import numpy as np
      return matrix_normal.rvs(mean = loc, rowcov = rowcov, colcov = colcov, size = nn, random_state=seed).astype(np.float32)
      # wrap python function as tensorflow op
    val = tf.py_func(np_sample, [self.loc, self.rowcov, self.colcov, nn], [tf.float32])[0]
    # set shape from unknown shape
    #import pdb; pdb.set_trace()
    #batch_event_shape = self.batch_shape.concatenate(self.event_shape)
    shape = tf.concat([tf.expand_dims(nn,0), tf.convert_to_tensor(self.loc.shape)], 0)
    val = tf.reshape(val, shape)
    return tf.convert_to_tensor(val)

    '''

    input_tensor = self.params
    if len(input_tensor.shape) == 0:
      input_tensor = tf.expand_dims(input_tensor, 0)
      multiples = tf.concat(
          [tf.expand_dims(n, 0), [1] * len(self.event_shape)], 0)
      return tf.tile(input_tensor, multiples)
    else:
      probs = tf.ones([self.n]) / tf.cast(self.n, dtype=tf.float32)
      cat = tf.contrib.distributions.Categorical(probs)
      indices = cat._sample_n(n, seed)
      tensor = tf.gather(input_tensor, indices)
      return tensor
    '''

# Generate random variable class similar to autogenerated ones from TensorFlow.
def __init__(self, *args, **kwargs):
  RandomVariable.__init__(self, *args, **kwargs)


_name = 'MatrixNormal'
_candidate = distributions_MatrixNormal
__init__.__doc__ = _candidate.__init__.__doc__
_globals = globals()
_params = {'__doc__': _candidate.__doc__,
           '__init__': __init__,
           'support': 'points'}
_globals[_name] = type(_name, (RandomVariable, _candidate), _params)
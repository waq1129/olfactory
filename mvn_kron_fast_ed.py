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
from tensorflow.contrib.distributions import Distribution, MultivariateNormalTriL

try:
  from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class distributions_MatrixNormal_Kron_fast(Distribution):
  """Matrix Variate random variable.
  #### Examples
  ```python
  # 100 samples of a scalar
  x = MVN(loc=tf.zeros(, rowcov, cov)
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
               cov_A=None,
               cov_B=None,
               cov_C=None,
               cov_D=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MatrixNormal_Kron_fast"):
    """Initialize a `Matrix Variate` random variable.
    Args:
      loc: 

      rowcov:

      cov:




      params: tf.Tensor.
      Collection of samples. Its outer (left-most) dimension
      determines the number of samples.

    ****how to keep track of batches covariance matrices


    """
    parameters = locals()

    self._loc = tf.convert_to_tensor(loc, name='loc')
    self._cov_A = tf.convert_to_tensor(cov_A, name='cov_A')
    self._cov_B = tf.convert_to_tensor(cov_B, name='cov_B')
    self._cov_C = tf.convert_to_tensor(cov_C, name='cov_C')
    self._cov_D = tf.convert_to_tensor(cov_D, name='cov_D')
    self._cov = tf.contrib.kfac.utils.kronecker_product(self._cov_A, self._cov_B)+tf.contrib.kfac.utils.kronecker_product(self._cov_C, self._cov_D)
    self._n = tf.shape(self.loc)[0]

    #parameters = {'loc': self.loc, 'rowcov': self.rowcov, 'cov': self.cov}


    '''
    with tf.name_scope(name, values=[params]):
      with tf.control_dependencies([]):
        self._params = tf.identity(params, name="params")
        try:
          self._n = tf.shape(self._params)[0]
        except ValueError:  # scalar params
          self._n = tf.constant(1)
    '''

    super(distributions_MatrixNormal_Kron_fast, self).__init__(
        dtype=self.loc.dtype,
        reparameterization_type=FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self.loc, self.cov_A, self.cov_B, self.cov_C, self.cov_D, self.n],
        name=name)

  @property
  def n(self):
    """Number of samples."""
    return self._n

  @property
  def loc(self):
    return self._loc
  
  @property
  def cov(self):
    return self._cov

  @property
  def cov_A(self):
    return self._cov_A

  @property
  def cov_B(self):
    return self._cov_B

  @property
  def cov_C(self):
    return self._cov_C

  @property
  def cov_D(self):
    return self._cov_D
  
  def _mean(self):
    return self._loc
  
  def _covariance(self):
    return self._cov

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
    import mvn_kron 
    val = mvn_kron.matnorm_logp(value, loc=self.loc, cov_A=self.cov_A, cov_B=self.cov_B, cov_C=self.cov_C, cov_D=self.cov_D)
    return val

  def _sample_n(self, nn, seed=None):
    mvn = MultivariateNormalTriL(loc=self.loc,scale_tril=tf.cholesky(self._cov))
    return mvn.sample(nn)

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


_name = 'MatrixNormal_Kron_fast'
_candidate = distributions_MatrixNormal_Kron_fast
__init__.__doc__ = _candidate.__init__.__doc__
_globals = globals()
_params = {'__doc__': _candidate.__doc__,
           '__init__': __init__,
           'support': 'points'}
_globals[_name] = type(_name, (RandomVariable, _candidate), _params)
import tensorflow as tf
import numpy as np
import abc
import scipy.linalg
import scipy.sparse
from tensorflow.contrib.distributions import InverseGamma, WishartCholesky
from helpers import define_scope, xx_t, scaled_I
from utils import tf_solve_lower_triangular_kron,\
                          tf_solve_upper_triangular_kron, \
                          tf_solve_lower_triangular_masked_kron, \
                          tf_solve_upper_triangular_masked_kron



def _mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col,
                         solve_row, solve_col):
    """Construct logp from the solves and determinants.
    """
    log2pi = 1.8378770664093453

    denominator = - rowsize * colsize * log2pi -\
        colsize * logdet_row - rowsize * logdet_col
    numerator = - tf.trace(tf.matmul(solve_col, solve_row))
    return 0.5 * (numerator + denominator)


def matnorm_logp(x, loc, row_cov, col_cov):
    """Log likelihood for centered matrix-variate normal density.
    Assumes that row_cov and col_cov follow the API defined in CovBase.
    """

    rowsize = tf.cast(tf.shape(x)[0], 'float32')
    colsize = tf.cast(tf.shape(x)[1], 'float32')
    x = tf.cast(x, tf.float32)
    colcovL = computeL(col_cov)
    solve_col = Sigma_inv_x(colcovL, tf.transpose(x-loc))
    logdet_col = logdet(colcovL)

    rowcovL = computeL(row_cov)
    solve_row = Sigma_inv_x(rowcovL, x-loc)
    logdet_row = logdet(rowcovL)

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                     logdet_col, solve_row, solve_col)


def computeL(Sigma):

    L = tf.cast(tf.cholesky(Sigma), tf.float32)
    #L = tf.matrix_set_diag(L, tf.log(tf.diag_part(L)))
    #L_full = tf.Variable(L, name="L_full", dtype="float64")
    #L_indeterminate = tf.matrix_band_part(L, -1, 0)
    #L = tf.matrix_set_diag(L_indeterminate, 
    #    tf.exp(tf.matrix_diag_part(L_indeterminate)))

    return L


def Sigma(self):
    """ covariance
    """
    return xx_t(self.L)

def get_optimize_vars(self):
    """ Returns a list of tf variables that need to get optimized to fit
         this covariance
    """
    return [self.L_full]

def logdet(L):
    """ log|Sigma| using a cholesky solve
    """
    return tf.cast(2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L))), tf.float32)

def Sigma_inv_x(L, X):
    """
    Given this Sigma and some X, compute :math:`Sigma^{-1} * x` using
    cholesky solve
    """
    return tf.cast(tf.cholesky_solve(tf.cast(L, tf.float64), tf.cast(X, tf.float64)), tf.float32)



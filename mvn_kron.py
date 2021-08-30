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



def _mnorm_logp_internal(size, logdet_, solve_, x):
    """Construct logp from the solves and determinants.
    """
    denominator = - size * np.log(2*np.pi) - logdet_
    numerator = - tf.reduce_sum(tf.multiply(solve_, tf.transpose(x)),0)
    return 0.5 * (numerator + denominator)


def matnorm_logp(x, loc, cov_A, cov_B, cov_C, cov_D):
    """Log likelihood for centered matrix-variate normal density.
    Assumes that row_cov and col_cov follow the API defined in CovBase.
    """

    size = tf.cast(tf.shape(x)[1], 'float32')
    nsample = tf.cast(tf.shape(x)[0], 'int32')
    x = tf.cast(x, tf.float32)
    d = tf.cast(tf.shape(x)[1], 'int32')
    
    s, u = my_eig(cov_A) 
    cov_AL = tf.matmul(u,tf.diag(tf.sqrt(tf.abs(s))))   
    s, u = my_eig(cov_B) 
    cov_BL = tf.matmul(u,tf.diag(tf.sqrt(tf.abs(s))))
    
    cov_ALinv = tf.linalg.inv(cov_AL)
    cov_BLinv = tf.linalg.inv(cov_BL)
    
    cc = tf.matmul(cov_ALinv,tf.matmul(cov_C,tf.transpose(cov_ALinv)))
    dd = tf.matmul(cov_BLinv,tf.matmul(cov_D,tf.transpose(cov_BLinv)))
    
    sc, uc = my_eig(cc) 
    sd, ud = my_eig(dd) 
    
    cdiag = tf.reshape(tf.matmul(tf.reshape(sc,(-1,1)),tf.reshape(sd,(1,-1))),[-1])
    cdiag = cdiag + tf.constant(1,dtype=tf.float32)
    cinv = tf.reciprocal(cdiag)
    cinvmat = repmat(cinv,nsample)
    
    lu = tf.matmul(cov_AL,uc)
    pv = tf.matmul(cov_BL,ud)
    luinv = tf.linalg.inv(lu)
    pvinv = tf.linalg.inv(pv)
    
    lupvinv_x = kronmult_AB_x(luinv,pvinv,tf.transpose(x-loc))
    cdiag_lupvinv_x = tf.multiply(cinvmat,lupvinv_x)
    solve_ = kronmult_AB_x(tf.transpose(luinv),tf.transpose(pvinv),cdiag_lupvinv_x)
    
    lluu = tf.matmul(tf.matmul(tf.transpose(cov_AL),cov_AL),tf.matmul(tf.transpose(uc),uc))
    ppvv = tf.matmul(tf.matmul(tf.transpose(cov_BL),cov_BL),tf.matmul(tf.transpose(ud),ud))
    logdet_ = logdet_AkronB(lluu,ppvv) + tf.reduce_sum(tf.log(cdiag))

#     cov_AL = tf.cholesky(cov_A)  
#     cov_ALinv = tf.linalg.inv(cov_AL)
#     cov_BL = tf.cholesky(cov_B)
#     cov_BLinv = tf.linalg.inv(cov_BL)

#     ab = tf.contrib.kfac.utils.kronecker_product(cov_AL, cov_BL)
#     abinv = tf.linalg.inv(ab)
#     Cov =  tf.matmul(ab, tf.transpose(ab))+tf.contrib.kfac.utils.kronecker_product(cov_C, cov_D)
#     #Cov =  tf.contrib.kfac.utils.kronecker_product(cov_A, cov_B)+tf.contrib.kfac.utils.kronecker_product(cov_C, cov_D)
#     #Kinv = tf.linalg.inv(Cov)
#     #CD = tf.contrib.kfac.utils.kronecker_product(cov_C, cov_D)
#     #kinv = tf.linalg.inv(tf.matmul(abinv,tf.matmul(CD,tf.transpose(abinv)))+tf.eye(d))
    
#     aCa = tf.matmul(cov_ALinv,tf.matmul(cov_C,tf.transpose(cov_ALinv)))
#     bDb = tf.matmul(cov_BLinv,tf.matmul(cov_D,tf.transpose(cov_BLinv)))

#     kinv = tf.linalg.inv(tf.contrib.kfac.utils.kronecker_product(aCa, bDb)+tf.eye(d))
    
#     #sc, uc = tf.self_adjoint_eig(aCa) 
#     #sd, ud = tf.self_adjoint_eig(bDb) 
    
#     sc = tf.diag_part(aCa)
#     sd = tf.diag_part(bDb)
#     #uc = tf.eye(tf.cast(tf.shape(aCa)[0], 'int32'))
#     #ud = tf.eye(tf.cast(tf.shape(bDb)[0], 'int32'))
#     uc = tf.eye(5)
#     ud = tf.eye(3)
#     print(sc)
#     print(sd)
#     print(uc)
#     print(ud)
    
#     uv = tf.contrib.kfac.utils.kronecker_product(uc,ud)
#     sl = tf.contrib.kfac.utils.kronecker_product(tf.diag(sc),tf.diag(sd))
#     kinv = tf.linalg.inv(tf.matmul(uv, tf.matmul(sl, tf.transpose(uv)))+tf.eye(d))
#     Kinv = tf.matmul(tf.transpose(abinv),tf.matmul(kinv,abinv))
    
#     solve_ = tf.matmul(Kinv, tf.transpose(x-loc))
#     #solve_ = Sigma_inv_x(covL, tf.transpose(x-loc))
#     #covL = computeL(Cov)
#     logdet_ = tf.linalg.logdet(Cov)
    
    return _mnorm_logp_internal(size, logdet_, solve_, x-loc)

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


def kronmult_AB_x(A,B,xx0):
    nrows = tf.cast(tf.shape(xx0)[0], 'int32')
    ncols = tf.cast(tf.shape(xx0)[1], 'int32')

    nA1 = tf.cast(tf.shape(A)[0], 'int32')
    nA2 = tf.cast(tf.shape(A)[1], 'int32')

    xx1 = tf.reshape(xx0,(nA2,-1))
    xx2 = tf.matmul(A,xx1)

    xx3 = tf.reshape(xx2,(nA1,tf.cast(nrows/nA2,'int32'),-1))
    xx4 = tf.transpose(xx3, [1, 0, 2])
    nrows = tf.cast(nA1*nrows/nA2,'int32')

    nB1 = tf.cast(tf.shape(B)[0], 'int32')
    nB2 = tf.cast(tf.shape(B)[1], 'int32')

    xx5 = tf.reshape(xx4,(nB2,-1))
    xx6 = tf.matmul(B,xx5)

    xx7 = tf.reshape(xx6,(nB1,tf.cast(nrows/nB2,'int32'),-1))
    xx8 = tf.transpose(xx7, [1, 0, 2])
    nrows = tf.cast(nB1*nrows/nB2,'int32')

    y = tf.reshape(xx8,(nrows,ncols))
    return y

def repmat(xx,m):
    n = tf.cast(tf.shape(xx)[0],'int32')
    xx1 = tf.tile(xx,[m])
    xx2 = tf.transpose(tf.reshape(xx1,(-1,n)))
    return xx2

def logdet_AkronB(A,B):
    rankA = tf.cast(tf.shape(A)[0],'float32')
    rankB = tf.cast(tf.shape(B)[0],'float32')
    detK = rankB*tf.linalg.logdet(A)+rankA*tf.linalg.logdet(B)
    return detK

def make_symmetric(A):
    return (A+tf.transpose(A))/2

def my_eig(cov_B):
    cov_BB = tf.reduce_sum(tf.abs(cov_B-tf.diag(tf.diag_part(cov_B))))
    pred = tf.less(cov_BB,1e-10)
    def val_if_true():
        sd = tf.diag_part(cov_B) 
        ud = tf.diag(tf.cast(tf.sign(tf.abs(tf.diag_part(cov_B))+10),'float32'))
        return sd, ud
    def val_if_false():
        sd, ud = tf.self_adjoint_eig(cov_B) 
        return sd, ud
    sd, ud = tf.cond(pred, val_if_true, val_if_false)
    return sd, ud

def solve_x(x, cov_A, cov_B, cov_C, cov_D):

    size = tf.cast(tf.shape(x)[1], 'float32')
    nsample = tf.cast(tf.shape(x)[0], 'int32')
    x = tf.cast(x, tf.float32)
    d = tf.cast(tf.shape(x)[1], 'int32')
    
    s, u = my_eig(cov_A) 
    cov_AL = tf.matmul(u,tf.diag(tf.sqrt(tf.abs(s))))   
    s, u = my_eig(cov_B) 
    cov_BL = tf.matmul(u,tf.diag(tf.sqrt(tf.abs(s))))
    
    cov_ALinv = tf.linalg.inv(cov_AL)
    cov_BLinv = tf.linalg.inv(cov_BL)
    
    cc = tf.matmul(cov_ALinv,tf.matmul(cov_C,tf.transpose(cov_ALinv)))
    dd = tf.matmul(cov_BLinv,tf.matmul(cov_D,tf.transpose(cov_BLinv)))
    
    sc, uc = my_eig(cc) 
    sd, ud = my_eig(dd) 
    
    cdiag = tf.reshape(tf.matmul(tf.reshape(sc,(-1,1)),tf.reshape(sd,(1,-1))),[-1])
    cdiag = cdiag + tf.constant(1,dtype=tf.float32)
    cinv = tf.reciprocal(cdiag)
    cinvmat = repmat(cinv,nsample)
    
    lu = tf.matmul(cov_AL,uc)
    pv = tf.matmul(cov_BL,ud)
    luinv = tf.linalg.inv(lu)
    pvinv = tf.linalg.inv(pv)
    
    lupvinv_x = kronmult_AB_x(luinv,pvinv,tf.transpose(x))
    cdiag_lupvinv_x = tf.multiply(cinvmat,lupvinv_x)
    solve_ = kronmult_AB_x(tf.transpose(luinv),tf.transpose(pvinv),cdiag_lupvinv_x)
    
    return solve_
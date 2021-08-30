from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np

def computeL(Sigma):
    L = np.linalg.cholesky(Sigma)
    return L

def logdet(L):
    """ log|Sigma| using a cholesky solve
    """
    return 2.0 * np.sum(np.log(np.diag(L)))

def Sigma_inv_x(L, X):
    """
    Given this Sigma and some X, compute :math:`Sigma^{-1} * x` using
    cholesky solve
    """
    Linv = np.linalg.inv(L)
    return np.dot(np.dot(Linv.T, Linv), X)

def make_symmetric(A):
    return (A+A.T)/2

def kronmult_AB_x(A,B,xx0):
    nrows = xx0.shape[0]
    ncols = xx0.shape[1]

    nA = A.shape[0]
    nB = B.shape[0]

    xx1 = np.reshape(xx0,(nA,-1))
    xx2 = np.dot(A, xx1)

    xx3 = np.reshape(xx2,(nA,nB,-1))
    xx4 = xx3.swapaxes(0,1)
    
    xx5 = np.reshape(xx4,(nB,-1))
    xx6 = np.dot(B, xx5)

    xx7 = np.reshape(xx6,(nB,nA,-1))
    xx8 = xx7.swapaxes(0,1)

    y = np.reshape(xx8,(nrows,ncols))
    return y

def logdet_AkronB(A,B):
    rankA = A.shape[0]
    rankB = B.shape[0]
    detK = rankB*logdet(computeL(A))+rankA*logdet(computeL(B))
    return detK

def my_eig(cov_B):
    db = cov_B.shape[0]
    diag_cov_B = np.tile(np.diag(cov_B),(db,1))
    cov_BB = np.sum(np.abs(cov_B-np.multiply(np.eye(db),diag_cov_B)))
    if cov_BB<1e-10:
        sd = np.diag(cov_B) 
        ud = np.eye(db)
    else:
        sd, ud = np.linalg.eigh(cov_B) 
    return sd, ud

def _mnorm_logp_internal(size, logdet_, solve_, x):
    """Construct logp from the solves and determinants.
    """
    denominator = - size * np.log(2*np.pi) - logdet_
    numerator = - np.sum(np.multiply(solve_, x.T), axis=0)
    return 0.5 * (numerator + denominator)

   
def matnorm_logp(x, loc, cov_A, cov_B, cov_C, cov_D):
    """
    log probability of matrix normal distribution whose equivalent 
    multivariate normal distribution has a mean at loc, and a covariance 
    is cov_A \kron cov_B + cov_C \kron cov_D
    """
    
    cov_A = make_symmetric(cov_A)
    cov_B = make_symmetric(cov_B)
    cov_C = make_symmetric(cov_C)
    cov_D = make_symmetric(cov_D)
    
    da = cov_A.shape[0]
    db = cov_B.shape[0]
    d = x.shape[1]
    nsample = x.shape[0]

    s, u = my_eig(cov_A) 
    cov_AL = np.multiply(u,np.tile(np.sqrt(np.abs(s)),(da,1))) 
    #cov_AL = computeL(cov_A)
    s, u = my_eig(cov_B) 
    cov_BL = np.multiply(u,np.tile(np.sqrt(np.abs(s)),(db,1))) 
    #cov_BL = computeL(cov_B)

    cov_ALinv = np.linalg.inv(cov_AL)
    cov_BLinv = np.linalg.inv(cov_BL)

    cc = np.dot(cov_ALinv , np.dot(cov_C, cov_ALinv.T))
    dd = np.dot(cov_BLinv , np.dot(cov_D, cov_BLinv.T))

    sc, uc = my_eig(cc) 
    sd, ud = my_eig(dd) 

    cdiag = np.dot(np.reshape(sc,(-1,1)), np.reshape(sd,(1,-1))).flatten()+1
    cinv = 1/cdiag
    cinvmat = np.tile(cinv,(nsample,1)).T

    lu = np.dot(cov_AL, uc)
    pv = np.dot(cov_BL, ud)
    luinv = np.linalg.inv(lu)
    pvinv = np.linalg.inv(pv)

    lupvinv_x = kronmult_AB_x(luinv,pvinv,(x-loc).T)
    cdiag_lupvinv_x = np.multiply(cinvmat,lupvinv_x)
    solve_ = kronmult_AB_x(luinv.T,pvinv.T,cdiag_lupvinv_x)

    lluu = np.dot(np.dot(cov_AL.T, np.dot(cov_AL, uc.T)), uc)
    ppvv = np.dot(np.dot(cov_BL.T, np.dot(cov_BL, ud.T)), ud)
    logdet_ = logdet_AkronB(lluu,ppvv) + np.sum(np.log(cdiag))

    return _mnorm_logp_internal(d, logdet_, solve_, x-loc)

def solve_x(x, cov_A, cov_B, cov_C, cov_D):
    """Log likelihood for centered matrix-variate normal density.
    Assumes that row_cov and col_cov follow the API defined in CovBase.
    """
    
    cov_A = make_symmetric(cov_A)
    cov_B = make_symmetric(cov_B)
    cov_C = make_symmetric(cov_C)
    cov_D = make_symmetric(cov_D)
    
    da = cov_A.shape[0]
    db = cov_B.shape[0]
    d = x.shape[1]
    nsample = x.shape[0]

    s, u = my_eig(cov_A) 
    cov_AL = np.multiply(u,np.tile(np.sqrt(np.abs(s)),(da,1))) 
    s, u = my_eig(cov_B) 
    cov_BL = np.multiply(u,np.tile(np.sqrt(np.abs(s)),(db,1))) 

    cov_ALinv = np.linalg.inv(cov_AL)
    cov_BLinv = np.linalg.inv(cov_BL)

    cc = np.dot(cov_ALinv , np.dot(cov_C, cov_ALinv.T))
    dd = np.dot(cov_BLinv , np.dot(cov_D, cov_BLinv.T))

    sc, uc = my_eig(cc) 
    sd, ud = my_eig(dd) 

    cdiag = np.dot(np.reshape(sc,(-1,1)), np.reshape(sd,(1,-1))).flatten()+1
    cinv = 1/cdiag
    cinvmat = np.tile(cinv,(nsample,1)).T

    lu = np.dot(cov_AL, uc)
    pv = np.dot(cov_BL, ud)
    luinv = np.linalg.inv(lu)
    pvinv = np.linalg.inv(pv)

    lupvinv_x = kronmult_AB_x(luinv,pvinv,x.T)
    cdiag_lupvinv_x = np.multiply(cinvmat,lupvinv_x)
    solve_ = kronmult_AB_x(luinv.T,pvinv.T,cdiag_lupvinv_x)

    return solve_

def matnorm_sample(loc,cov_A,cov_B,cov_C,cov_D,T):
    """
    generate sample from matrix normal distribution whose equivalent 
    multivariate normal distribution has a mean at loc, and a covariance 
    is cov_A \kron cov_B + cov_C \kron cov_D
    """
        
    cov_A = make_symmetric(cov_A)
    cov_B = make_symmetric(cov_B)
    cov_C = make_symmetric(cov_C)
    cov_D = make_symmetric(cov_D)
    da = cov_A.shape[0]
    db = cov_B.shape[0]
    N = da
    D = db
    
    s, u = my_eig(cov_A) 
    cov_AL = np.multiply(u,np.tile(np.sqrt(np.abs(s)),(da,1))) 
    s, u = my_eig(cov_B) 
    cov_BL = np.multiply(u,np.tile(np.sqrt(np.abs(s)),(db,1))) 

    cov_ALinv = np.linalg.inv(cov_AL)
    cov_BLinv = np.linalg.inv(cov_BL)

    cc = cov_ALinv @ cov_C @ cov_ALinv.T
    dd = cov_BLinv @ cov_D @ cov_BLinv.T

    sc, uc = my_eig(cc) 
    sd, ud = my_eig(dd) 

    cdiag = (np.reshape(sc,(-1,1)) @ np.reshape(sd,(1,-1))).flatten()+1
    cdiag = np.sqrt(np.abs(cdiag))

    lu = cov_AL @ uc
    pv = cov_BL @ ud

    norm = np.random.randn(T,N*D)
    s01cdiag = np.multiply(norm, np.tile(cdiag,(T,1)))
    ss = kronmult_AB_x(lu,pv,s01cdiag.T).T+loc.flatten()
    return ss  
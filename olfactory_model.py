from olfactory_import import *
from olfactory_util import *

def negative_ll_A(x_true_trial,Kernel,sig_o,D,N):
    T = x_true_trial.shape[0]
    A = x_true_trial.T
    a_cumsum = np.cumsum(A,1)
    c1 = 1/np.sqrt(np.multiply(np.linspace(1,T-1,T-1), np.linspace(2,T,T-1)))
    c1 = np.hstack((c1, 1/np.sqrt(T)))
    c2 = np.sqrt(np.divide(np.linspace(1,T-1,T-1), np.linspace(2,T,T-1)))
    c2 = np.hstack((0, c2))
    a1 = np.multiply(np.tile(c1,(D*N,1)),a_cumsum)
    a2 = np.roll(np.multiply(np.tile(c2,(D*N,1)),A),-1)
    x_m = a1-a2
    x_m_T_1 = x_m[:,:-1]
    x_m_T = x_m[:,-1].reshape(N,D)

    x1 = MultivariateNormalTriL(loc=tf.zeros((D*(T-1),N)), scale_tril=tf.cholesky(sig_o))
    x2 = MultivariateNormalTriL(loc=tf.zeros((D,N)), scale_tril=tf.cholesky(Kernel*T+sig_o))
    x1t = x_m_T_1.T.reshape(T-1,N,D).swapaxes(1,2).reshape(-1,N)
    x2t = x_m_T.T
    
    return x1, x2, x1t, x2t

def negative_ll_sign_B2(x_true_trial,Kernel,sig_o,sig_n,D,N):
    T = x_true_trial.shape[0]
    A = x_true_trial.T
    a_cumsum = np.cumsum(A,1)
    c1 = 1/np.sqrt(np.multiply(np.linspace(1,T-1,T-1), np.linspace(2,T,T-1)))
    c1 = np.hstack((c1, 1/np.sqrt(T)))
    c2 = np.sqrt(np.divide(np.linspace(1,T-1,T-1), np.linspace(2,T,T-1)))
    c2 = np.hstack((0, c2))
    a1 = np.multiply(np.tile(c1,(D*N,1)),a_cumsum)
    a2 = np.roll(np.multiply(np.tile(c2,(D*N,1)),A),-1)
    x_m = a1-a2
    x_m_T_1 = x_m[:,:-1]
    x_m_T = x_m[:,-1].reshape(-1,1)
    
    x1 = MatrixNormal_Kron_fast_sample(loc=const(np.zeros([T-1, D*N])), cov_C=tf.eye(N)*0, cov_D=tf.eye(D)*0, 
                                      cov_A=sig_o, cov_B=sig_n)
    x2 = MatrixNormal_Kron_fast_sample(loc=const(np.zeros([1, D*N])), cov_C=Kernel, cov_D=tf.eye(D)*T, 
                                      cov_A=sig_o, cov_B=sig_n)
    x1t = x_m_T_1.T
    x2t = x_m_T.T
    
    return x1, x2, x1t, x2t

def pack_params(z_init_tr,sig_f1_z_est,l_z1_est,l_z_est,Bnest,Boest,l_B_nest,l_B_oest,sig_o1_z_est,sig_n1_z_est,model_flag):
    if model_flag==1:
        sig_o1_z = sig_o1_z_est
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_o1_z))
    if model_flag==2:
        sig_n1_z = sig_n1_z_est
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_n1_z))
    if model_flag==3:
        sig_o1_z = sig_o1_z_est
        sig_n1_z = sig_n1_z_est
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_o1_z, sig_n1_z))
    if model_flag==4:
        sig_o1_z = sig_o1_z_est
        sig_n1_z = sig_n1_z_est
        Bn = Bnest
        l_B_n = l_B_nest
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_o1_z, sig_n1_z, Bn.flatten(), l_B_n))
    if model_flag==5:
        sig_o1_z = sig_o1_z_est
        sig_n1_z = sig_n1_z_est
        Bo = Boest
        l_B_o = l_B_oest
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_o1_z, sig_n1_z, Bo.flatten(), l_B_o))
    if model_flag==6:
        sig_n1_z = sig_n1_z_est
        sig_o1_z = sig_o1_z_est
        Bo = Boest
        l_B_o = l_B_oest
        Bn = Bnest
        l_B_n = l_B_nest
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_o1_z, sig_n1_z, \
                          Bo.flatten(), l_B_o, Bn.flatten(), l_B_n))
    if model_flag==7:
        sig_o1_z = sig_o1_z_est
        Bo = Boest
        l_B_o = l_B_oest
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_o1_z, Bo.flatten(), l_B_o))
    if model_flag==8:
        sig_n1_z = sig_n1_z_est
        Bn = Bnest
        l_B_n = l_B_nest
        return np.hstack((z_init_tr.flatten(), sig_f1_z_est, l_z1_est, l_z_est, sig_n1_z, Bn.flatten(), l_B_n))

def unpack_params(params,N1_tr,D1,K,R,model_flag):
    z_init_tr = params[:N1_tr*K].reshape(N1_tr,K)
    sig_f = params[N1_tr*K:N1_tr*K+1].flatten()
    lz1 = params[N1_tr*K+1:N1_tr*K+1+1].flatten()
    lz = params[N1_tr*K+1+1:N1_tr*K+1+1+K].flatten()
    ss = N1_tr*K+1+1+K
    if model_flag==1:
        sig_o1_z = params[ss:ss+N1_tr].flatten()
        sig_n1_z = np.ones(D1)
        Bo = np.zeros((N1_tr,R))
        l_B_o = np.zeros(R)
        Bn = np.zeros((D1,R))
        l_B_n = np.zeros(R)
    if model_flag==2:
        sig_n1_z = params[ss:ss+D1].flatten()
        sig_o1_z = np.ones(N1_tr)
        Bo = np.zeros((N1_tr,R))
        l_B_o = np.zeros(R)
        Bn = np.zeros((D1,R))
        l_B_n = np.zeros(R)
    if model_flag==3:
        sig_o1_z = params[ss:ss+N1_tr].flatten()
        sig_n1_z = params[ss+N1_tr:ss+N1_tr+D1].flatten()
        Bo = np.zeros((N1_tr,R))
        l_B_o = np.zeros(R)
        Bn = np.zeros((D1,R))
        l_B_n = np.zeros(R)
    if model_flag==4:
        sig_o1_z = params[ss:ss+N1_tr].flatten()
        sig_n1_z = params[ss+N1_tr:ss+N1_tr+D1].flatten()
        Bn = params[ss+N1_tr+D1:ss+N1_tr+D1+R*D1].reshape(D1,R)
        l_B_n = params[ss+N1_tr+D1+R*D1:ss+N1_tr+D1+R*D1+R].flatten()
        Bo = np.zeros((N1_tr,R))
        l_B_o = np.zeros(R)
    if model_flag==5:
        sig_o1_z = params[ss:ss+N1_tr].flatten()
        sig_n1_z = params[ss+N1_tr:ss+N1_tr+D1].flatten()
        Bo = params[ss+N1_tr+D1:ss+N1_tr+D1+R*N1_tr].reshape(N1_tr,R)
        l_B_o = params[ss+N1_tr+D1+R*N1_tr:ss+N1_tr+D1+R*N1_tr+R].flatten()
        Bn = np.zeros((D1,R))
        l_B_n = np.zeros(R)
    if model_flag==6:
        sig_n1_z = params[ss+N1_tr:ss+N1_tr+D1].flatten()
        sig_o1_z = params[ss:ss+N1_tr].flatten()
        Bo = params[ss+N1_tr+D1:ss+N1_tr+D1+R*N1_tr].reshape(N1_tr,R)
        l_B_o = params[ss+N1_tr+D1+R*N1_tr:ss+N1_tr+D1+R*N1_tr+R].flatten()
        Bn = params[ss+N1_tr+D1+R*N1_tr+R:ss+N1_tr+D1+R*N1_tr+R+R*D1].reshape(D1,R)
        l_B_n = params[ss+N1_tr+D1+R*N1_tr+R+R*D1:ss+N1_tr+D1+R*N1_tr+R+R*D1+R].flatten()
    if model_flag==7:
        sig_o1_z = params[ss:ss+N1_tr].flatten()
        sig_n1_z = np.ones(D1)
        Bo = params[ss+N1_tr:ss+N1_tr+R*N1_tr].reshape(N1_tr,R)
        l_B_o = params[ss+N1_tr+R*N1_tr:ss+N1_tr+R*N1_tr+R].flatten()
        Bn = np.zeros((D1,R))
        l_B_n = np.zeros(R)
    if model_flag==8:
        sig_n1_z = params[ss:ss+D1].flatten()
        sig_o1_z = np.ones(N1_tr)
        Bn = params[ss+D1:ss+D1+R*D1].reshape(D1,R)
        l_B_n = params[ss+D1+R*D1:ss+D1+R*D1+R].flatten()
        Bo = np.zeros((N1_tr,R))
        l_B_o = np.zeros(R)

    return z_init_tr, sig_f, lz1, lz, sig_o1_z, sig_n1_z, Bo, l_B_o, Bn, l_B_n

def tmp(A,n,o):
    T = A.shape[1]
    a_cumsum = np.cumsum(A,1)
    c1 = 1/np.sqrt(np.multiply(np.linspace(1,T-1,T-1), np.linspace(2,T,T-1)))
    c1 = np.hstack((c1, 1/np.sqrt(T)))
    c2 = np.sqrt(np.divide(np.linspace(1,T-1,T-1), np.linspace(2,T,T-1)))
    c2 = np.hstack((0, c2))
    a1 = np.multiply(np.tile(c1,(n*o,1)),a_cumsum)
    a2 = np.roll(np.multiply(np.tile(c2,(n*o,1)),A),-1)

    x_m = a1-a2
    x_m_T_1 = x_m[:,:-1]
    x_m_T = x_m[:,-1].reshape(-1,1)
    return x_m_T_1, x_m_T

def negative_ll_tr(params,a1_est,b1_est,N1_tr,D1,T,K,R,model_flag,kernel_flag,trial_flag,x_true_avg1_train,x_true1_trial_train):
    z_init_tr, sig_f, lz1, lz, sig_o1_z, sig_n1_z, Bo, l_B_o, Bn, l_B_n = unpack_params(params,N1_tr,D1,K,R,model_flag)
    z1 = np.dot(z_init_tr,np.diag(np.sqrt(np.exp(lz))))
    if kernel_flag=='rbf':
        K_z1 = rbf_covariance(np.hstack((sig_f,lz1)),z1,z1)
    elif kernel_flag=='linear':
        K_z1 = np.dot(z1, z1.T)
    elif kernel_flag=='mixture':
        k_rbf1 = rbf_covariance(np.hstack((sig_f,lz1)),z1,z1)
        k_linear1 = np.dot(z1, z1.T)
        K_z1 = k_rbf1*b1_est+k_linear1*a1_est

    # data likelihood
    B_n = np.dot(np.dot(Bn,np.diag(np.exp(l_B_n))),Bn.T)+np.diag(np.exp(sig_n1_z))
    B_o = np.dot(np.dot(Bo,np.diag(np.exp(l_B_o))),Bo.T)+np.diag(np.exp(sig_o1_z))

    if trial_flag==0:
        if model_flag==1 or model_flag==7:
            Sigma = K_z1+B_o
            Sigma_L = np.linalg.cholesky(Sigma)
            Sigma_Linv = np.linalg.inv(Sigma_L)
            Sigmainv = np.dot(Sigma_Linv.T, Sigma_Linv)

            Lambda = np.dot(x_true_avg1_train, x_true_avg1_train.T)
            ll = -0.5*np.trace(np.dot(Sigmainv, Lambda)) - 0.5*D1*logdet(Sigma_L)-0.5*np.trace(np.dot(z_init_tr.T, z_init_tr))
        else:
            loc = np.zeros((1,D1*N1_tr))
            ll = np.sum(mvn_kron_autograd.matnorm_logp(x_true_avg1_train.reshape(1,-1),loc,\
                                                       cov_A=B_o,cov_B=B_n,cov_C=K_z1,cov_D=np.eye(D1)))
    else:
        x_m_T_1, x_m_T = tmp(x_true1_trial_train.T,D1,N1_tr)
        if model_flag==1 or model_flag==7:
            Sigma = B_o
            Sigma_L = np.linalg.cholesky(Sigma)
            Sigma_Linv = np.linalg.inv(Sigma_L)
            Sigmainv = np.dot(Sigma_Linv.T, Sigma_Linv)

            x1t = x_m_T_1.T.reshape(T-1,N1_tr,D1).swapaxes(1,2).reshape(-1,N1_tr)
            Lambda = np.dot(x1t.T, x1t)
            ll1 = -0.5*np.trace(np.dot(Sigmainv, Lambda)) - 0.5*D1*(T-1)*logdet(Sigma_L)

            Sigma = K_z1*T+B_o
            Sigma_L = np.linalg.cholesky(Sigma)
            Sigma_Linv = np.linalg.inv(Sigma_L)
            Sigmainv = np.dot(Sigma_Linv.T, Sigma_Linv)

            Lambda = np.dot(x_m_T.reshape(N1_tr,D1), x_m_T.reshape(N1_tr,D1).T)
            ll2 = -0.5*np.trace(np.dot(Sigmainv, Lambda)) - 0.5*D1*logdet(Sigma_L)

            ll = ll1+ll2-0.5*np.trace(np.dot(z_init_tr.T, z_init_tr))
        else:
            loc = np.zeros((T-1,D1*N1_tr))
            ll1 = np.sum(mvn_kron_autograd.matnorm_logp(x_m_T_1.T,loc,\
                                                       cov_A=B_o,cov_B=B_n,cov_C=np.eye(N1_tr)*0,cov_D=np.eye(D1)*0))
            loc = np.zeros((1,D1*N1_tr))
            ll2 = np.sum(mvn_kron_autograd.matnorm_logp(x_m_T.T,loc,\
                                                       cov_A=B_o,cov_B=B_n,cov_C=K_z1,cov_D=np.eye(D1)*T))
            ll = ll1+ll2-0.5*np.trace(np.dot(z_init_tr.T, z_init_tr))

    return -ll

def negative_ll_te(params,qz_tr_mean1,a1_est,b1_est,sig_f,lz1,lz,sig_o1_z,sig_n1_z,Bo,l_B_o,Bn,l_B_n,N1_tr,N1_te,D1,D1_va,T,K,R,model_flag,kernel_flag,trial_flag,x_true_avg1_test_va_all,x_true1_trial_test_va_all,n1ii):
    
    z_init_te = params[:N1_te*K].reshape(N1_te,K)
    z_all = np.vstack((qz_tr_mean1,z_init_te))
    z1 = np.dot(z_all,np.diag(np.sqrt(np.exp(lz))))
    
    if kernel_flag=='rbf':
        K_z1 = rbf_covariance(np.hstack((sig_f,lz1)),z1,z1)
    elif kernel_flag=='linear':
        K_z1 = np.dot(z1, z1.T)
    elif kernel_flag=='mixture':
        k_rbf1 = rbf_covariance(np.hstack((sig_f,lz1)),z1,z1)
        k_linear1 = np.dot(z1, z1.T)
        K_z1 = k_rbf1*b1_est+k_linear1*a1_est
         
    # data likelihood
    B_n = np.dot(np.dot(Bn,np.diag(np.exp(l_B_n))),Bn.T)+np.diag(np.exp(sig_n1_z))
    B_n1 = B_n[n1ii,:]
    B_n1 = B_n1[:,n1ii]
    B_o = np.dot(np.dot(Bo,np.diag(np.exp(l_B_o))),Bo.T)+np.diag(np.exp(sig_o1_z))
    
    if model_flag==2 or model_flag==8:
        sig_o1_z_va = np.zeros(N1_te)
    else:
        sig_o1_z_va = np.exp(params[-1])
    sig_o1_z_all = np.hstack((sig_o1_z,sig_o1_z_va))

    Bo_va = params[N1_te*K:N1_te*K+N1_te*R].reshape(N1_te,R)
    Bo_all = np.vstack((Bo,Bo_va))
    if model_flag==5 or model_flag==6 or model_flag==7:
        B_o_all = np.dot(Bo_all,np.dot(np.diag(np.exp(l_B_o)),Bo_all.T))+np.diag(np.exp(sig_o1_z_all))
    else:
        B_o_all = np.diag(np.exp(sig_o1_z_all))


    if trial_flag==0:
        if model_flag==1 or model_flag==7:
            Sigma = K_z1+B_o_all
            Sigma_L = np.linalg.cholesky(Sigma)
            Sigma_Linv = np.linalg.inv(Sigma_L)
            Sigmainv = np.dot(Sigma_Linv.T, Sigma_Linv)

            Lambda = np.dot(x_true_avg1_test_va_all, x_true_avg1_test_va_all.T)
            ll = -0.5*np.trace(np.dot(Sigmainv, Lambda)) - 0.5*D1_va*logdet(Sigma_L)-0.5*np.trace(np.dot(z_all.T, z_all))
        else:
            loc = np.zeros((1,D1_va*(N1_tr+N1_te)))
            ll = np.sum(mvn_kron_autograd.matnorm_logp(x_true_avg1_test_va_all.reshape(1,-1),loc,\
                                                       cov_A=B_o_all,cov_B=B_n1,cov_C=K_z1,cov_D=np.eye(D1_va)))
    else:
        x_m_T_1, x_m_T = tmp(x_true1_trial_test_va_all.T,D1_va,N1_tr+N1_te)
        if model_flag==1 or model_flag==7:
            Sigma = B_o_all
            Sigma_L = np.linalg.cholesky(Sigma)
            Sigma_Linv = np.linalg.inv(Sigma_L)
            Sigmainv = np.dot(Sigma_Linv.T, Sigma_Linv)

            x1t = x_m_T_1.T.reshape(T-1,N1_tr+N1_te,D1_va).swapaxes(1,2).reshape(-1,N1_tr+N1_te)
            Lambda = np.dot(x1t.T, x1t)
            ll1 = -0.5*np.trace(np.dot(Sigmainv, Lambda)) - 0.5*D1_va*(T-1)*logdet(Sigma_L)

            Sigma = K_z1*T+B_o_all
            Sigma_L = np.linalg.cholesky(Sigma)
            Sigma_Linv = np.linalg.inv(Sigma_L)
            Sigmainv = np.dot(Sigma_Linv.T, Sigma_Linv)

            Lambda = np.dot(x_m_T.reshape(N1_tr+N1_te,D1_va), x_m_T.reshape(N1_tr+N1_te,D1_va).T)
            ll2 = -0.5*np.trace(np.dot(Sigmainv, Lambda)) - 0.5*D1_va*logdet(Sigma_L)

            ll = ll1+ll2-0.5*np.trace(np.dot(z_all.T, z_all))
        else:
            loc = np.zeros((T-1,D1_va*(N1_tr+N1_te)))
            ll1 = np.sum(mvn_kron_autograd.matnorm_logp(x_m_T_1.T,loc,cov_A=B_o_all,cov_B=B_n1,\
                                                        cov_C=np.eye(N1_tr+N1_te)*0,cov_D=np.eye(D1_va)*0))
            loc = np.zeros((1,D1_va*(N1_tr+N1_te)))
            ll2 = np.sum(mvn_kron_autograd.matnorm_logp(x_m_T.T,loc,\
                                                       cov_A=B_o_all,cov_B=B_n1,cov_C=K_z1,cov_D=np.eye(D1_va)*T))
            ll = ll1+ll2-0.5*np.trace(np.dot(z_all.T, z_all))

    return -ll

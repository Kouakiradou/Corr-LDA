#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from numpy import matlib
from scipy.special import digamma, polygamma
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def simulation_data(D=50, k=10, V=1000, xi=40, max_iter=100, gamma_shape=2, gamma_scale=1):
    """Simulation the data according to LDA process. Return a list
    of N_d*V matrix(one-hot-coding) with length M.
    --------------------------------------------------------------
    Input:
    number of documents D,
    number of topics k,
    number of vocabulary V,
    the parameter xi for possion distribution xi (generate the length of each document),
    the parameter gamma_shape,gamma_scale for gamma distribution (generate the alpha in the paper);
    ---------------------------------------------------------------
    Output:

    images: vectorized images. a length N list of Nd*5 matrix, with each line to be a vector of a image region,
    captions: word documents. a length M list of Md*V matrix (one-hot-coding),
    alpha : concentrate parameters for dirichlet distribution, k*1 vector,
    BETA: k*V matrix.
    Mean: k*5 matrix.
    Covariance: k*5 matrix

    ** Assume that the length of vector of a image region is 5.
    """

    images = []
    captions = []

    # hyperparameter
    alpha = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=k)
    BETA = np.random.dirichlet(np.ones(V), k)
    Mean = np.random.normal(0, 1, (k, 5))
    # Covariance = np.random.normal(1, 1, (k, 5))
    Covariances = []
    for i in range(k):
        temp_mat = np.random.rand(5,5)
        Covariances.append(temp_mat @ temp_mat.T)

    # image level
    N = np.random.poisson(lam=(xi - 1), size=D) + 1  # number of regions in a image, avoid 0 words
    M = np.random.poisson(lam=(xi - 1), size=D) + 1  # number of words of captions in the image
    THETA = np.random.dirichlet(alpha, D)

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N[d])
        Y = np.random.randint(0, N[d], M[d])
        temp_mean = Z @ Mean
        indexes = np.arange(Z.shape[1]) @ Z.T
        # temp_covar = Z @ Covariance
        R = np.zeros((N[d], 5))
        W = np.zeros((M[d], V))
        for n in range(N[d]):
            covar_mat = Covariances[indexes[n]]
            R[n,] = np.random.multivariate_normal(temp_mean[n,], covar_mat)
        images.append(R)
        for m in range(M[d]):
            temp_BETA = Z[Y[m]] @ BETA  # we can actually manipulate the matrix in one step, fix it in the future
            W[m,] = np.random.multinomial(1, temp_BETA)
        captions.append(W)
    return images, captions, alpha, BETA, Mean, Covariances


def E_step_Vectorization(alpha, BETA, Mean, Covariances, image, caption, Phi0, gamma0, Lambda0, max_iter=100, tol=1e-4):
    # print("new doc")
    """
    Vectorization Version Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    BETA as a k*V matrix;
    Mean as a k*5 matrix.
    Covariance as a k*5 matrix
    image as a Nd*5 matrix;
    captions as a Md*V matrix
    Phi0 as a Nd*k matrix;
    gamma0 as a k*1 vector;
    Lambda0 is a Md*N matrix;
    tol as a float: tolerance.
    -------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma."""

    # Initialization
    Phi = Phi0
    gamma = gamma0
    Lambdaa = Lambda0
    phi_delta = 1
    gamma_delta = 1
    Lambdaa_delta = 1

    # relative tolerance is for each element in the matrix
    tol = tol ** 2

    for iteration in range(max_iter):
        ##update gamma
        gamma = alpha + Phi.sum(axis=0)
        # if not np.all(gamma >= 0):
        #     print(gamma)
        multivari_pdf = np.zeros((image.shape[0], Mean.shape[0]))
        for k in range(Mean.shape[0]):
            # print(np.diag(Covariance[k,]))
            # print(Mean[0])
            temp_pdf = multivariate_normal(mean=Mean[k], cov=Covariances[k]).pdf(image)  # Nd*1
            multivari_pdf[:, k] = temp_pdf

        ##update Phi
        # dia_gamma = np.exp(digamma(gamma) - digamma(sum(gamma)))
        # print(dia_gamma)
        # print(gamma)
        # print(digamma(gamma) - digamma(sum(gamma)))
        #----------------------------------------------
        # if 0 in dia_gamma:
        #     print(Phi)
        #     print(alpha)
        #     print("zero in dia_gamma")
        #     print(gamma)
        #     print(digamma(gamma) - digamma(sum(gamma)))
        #     print(dia_gamma(gamma))
        #     print(digamma(sum(gamma)))
        #     print(dia_gamma)
        # log_likelihood = np.exp(Lambdaa.T @ np.log(caption @ BETA.T))
        Phi = multivari_pdf * np.exp(digamma(gamma) - digamma(sum(gamma))) * np.exp(Lambdaa.T @ np.log(caption @ BETA.T)) #please double check
        # dia_gamma = np.exp(digamma(gamma) - digamma(sum(gamma)))
        # expppp = np.exp(Lambdaa.T @ np.log(caption @ BETA.T))
        # exp = np.exp(Lambdaa.T @ np.log(caption @ BETA.T))
        # diag = np.exp(digamma(gamma) - digamma(sum(gamma)))
        # print("-----Phi-------")
        # print(Phi)
        # print("-----Phi sum-------")
        # print(Phi.sum(axis=1)[:, None])
        # print("-----pdf-------")
        # print(multivari_pdf)
        # print("-----diagamma-------")
        # print(np.exp(digamma(gamma) - digamma(sum(gamma))))
        # print("-----exp-------")
        # print(np.exp(Lambdaa.T @ (caption @ BETA.T)))
        # print(Lambdaa.T)
        # print((caption @ BETA.T))
        # print("-----mean-------")
        # print(Mean)
        # print("-----covariances-------")
        # print(Covariances)
        # print("-----image-------")
        # print(image)


        Phi = Phi / (Phi.sum(axis=1)[:, None])  # row sum to 1

        ##update Lambda
        Lambdaa = np.exp(np.log(caption @ BETA.T) @ Phi.T)
        Lambdaa = Lambdaa / (Lambdaa.sum(axis=1)[:, None])

        # Phi = (doc @ BETA.T) * np.exp(digamma(gamma) - digamma(sum(gamma)))
        # Phi = Phi / (Phi.sum(axis=1)[:, None])  # row sum to 1

        # ##update gamma
        # gamma = alpha + Phi.sum(axis=0)

        ##check the convergence
        phi_delta = np.mean((Phi - Phi0) ** 2)
        gamma_delta = np.mean((gamma - gamma0) ** 2)
        Lambdaa_delta = np.mean((Lambdaa - Lambda0) ** 2)

        ##refill
        Phi0 = Phi
        gamma0 = gamma
        Lambda0 = Lambdaa
        # if phi_delta <= tol:
        #     print("phi tol")
        #
        if gamma_delta <= tol:
            print(gamma_delta)
        #
        # if Lambdaa_delta <= tol:
        #     print("lambda tol")
        if (phi_delta <= tol) and (gamma_delta <= tol) and (Lambdaa_delta <= tol):
            break

    return Phi, gamma, Lambdaa


def  M_step_Vectorization(images, captions, k, tol=1e-3, tol_estep=1e-3, max_iter=100, initial_alpha_shape=100,
                         initial_alpha_scale=0.01):
    """
    Vectorization version VI EM for Latent Dirichlet Allocation: M-step.
    Do to a list of documnents. -- a list of matrix.
    -------------------------------------------------
    Input:
    docs: a list of one-hot-coding matrix ;
    k: a fixed positive integer indicate the number of topics;
    tol,tol_estep: tolerance for Mstep,Estep;
    max_iter:max iteration for E-step, M-step;
    inital_alpha_shape,scale: initial parameters for alpha. (Parameters for gamma distribution)
    -------------------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma;
    optimal k*V matrix BETA;
    optimal k*1 vector alpha.
    """

    # get basic iteration
    D = len(images)
    V = captions[1].shape[1]
    N = [image.shape[0] for image in images]
    M = [caption.shape[0] for caption in captions]
    dim = images[0].shape[1]

    # initialization
    BETA0 = np.random.dirichlet(np.ones(V), k)
    alpha0 = np.random.gamma(shape=initial_alpha_shape, scale=initial_alpha_scale, size=k)
    print("initial alpha is: ", alpha0)
    Mean0 = np.random.normal(0, 1, (k, dim))

    Covariances0 = []
    for i in range(k):
        # temp_mat = np.random.rand(dim, dim)
        # Covariances0.append(temp_mat @ temp_mat.T)
        # Covariances0.append(make_spd_matrix(dim))
        Covariances0.append(np.eye(dim))

    PHI = [np.ones((N[d], k)) / k for d in range(D)]
    LAMBDA = [np.ones((M[d], N[d])) / N[d] for d in range(D)]
    GAMMA = np.array([alpha0 + N[d] / k for d in range(D)]) #?? why use N[d] but bot M[d]

    # Nk = np.zeros(k)

    # BETA = BETA0
    # alpha = alpha0
    # Mean = Mean0
    # Covariances = Covariances0
    alpha_dis = 1
    beta_dis = 1

    tol = tol ** 2

    for iteration in range(max_iter):
        # update PHI,GAMMA,BETA
        Nk = np.zeros(k)
        BETA = np.zeros((k, V)) + 1e-10
        for d in range(D):  #documents
            PHI[d], GAMMA[d,], LAMBDA[d] = E_step_Vectorization(alpha0, BETA0, Mean0, Covariances0, images[d], captions[d], PHI[d], GAMMA[d,], LAMBDA[d], max_iter, tol_estep)
            BETA += (LAMBDA[d] @ PHI[d]).T @ captions[d]
            Nk += np.sum(PHI[d], axis=0)  #calculating the Nk for EM on mean and covariance
        BETA = BETA / (BETA.sum(axis=1)[:, None])  # rowsum=1

        # update alpha
        z = D * polygamma(1, sum(alpha0))
        h = -D * polygamma(1, alpha0)
        g = D * (digamma(sum(alpha0)) - digamma(alpha0)) + (digamma(GAMMA) - digamma(GAMMA.sum(axis=1))[:, None]).sum(
            axis=0)
        c = (sum(g / h)) / (1 / z + sum(1 / h))
        alpha = alpha0 - (g - c) / h
        alpha[alpha < 0.01] = 0.01

        #update mean and covariance
        Mean = np.mat(np.zeros((k, dim)))
        # Covariances = [np.zeros((dim, dim))] * k # that is a fucking big error, remember this create k shallow copy of matrix
        Covariances = [np.zeros((dim, dim)) for _ in range(k)]
        for d in range(D):
            for i in range(k):
                Mean[i, :] += np.sum(np.multiply(images[d], np.mat(PHI[d][:, i]).T), axis=0) / Nk[i]
                cov_k = (images[d] - Mean[i]).T * np.multiply((images[d] - Mean[i]), np.mat(PHI[d][:, i]).T) / Nk[i]
                Covariances[i] += cov_k
        for i in range(k):
            Covariances[i] += np.identity(dim) * 0.001
        alpha_dis = np.mean((alpha - alpha0) ** 2)
        beta_dis = np.mean((BETA - BETA0) ** 2)
        mean_dis = np.mean((np.asarray(Mean) - Mean0) ** 2)
        cov_dis = np.mean((np.asarray(Covariances)- np.asarray(Covariances0)) ** 2)
        alpha0 = alpha
        BETA0 = BETA
        Mean0 = np.asarray(Mean)
        Covariances0 = Covariances
        if ((alpha_dis <= tol) and (beta_dis <= tol) and (mean_dis <= tol) and (cov_dis <= tol)): # check all the parameter
            break
        print(iteration)
    return alpha, BETA, Mean, Covariances, PHI, LAMBDA, GAMMA


def mmse(alpha, BETA, Mean, alpha_est, BETA_est, Mean_est):
    """
    Calculate mse for alpha and BETA . Input the true and estimate
    alpha,BETA .
    -------------------------------------------------
    Input:
    true alpha -- vector;
    true BETA -- matrix;
    estimator alpha -- vector;
    estimator BETA_est -- matrix;
    -------------------------------------------------
    Output:
    MSE defined in the report.
    """
    alpha_norm = alpha / np.sum(alpha)
    beta_mse = np.mean((BETA_est - BETA) ** 2)
    alpha_est_norm = alpha_est / np.sum(alpha_est)
    alpha_mse = np.mean((alpha_est_norm - alpha_norm) ** 2)
    Mean_mse = np.mean(np.multiply((Mean_est - Mean), (Mean_est - Mean)))
    return alpha_mse, beta_mse, Mean_mse

# docs, alpha, BETA = simulation_data2()
# alpha_est, beta_est = M_step_Vectorization(docs=docs,k=10,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)
# alpha_mse, beta_mse = mmse(alpha, BETA, alpha_est, beta_est)
# print(alpha_mse, beta_mse)

'''
one case of multivariate distribution 
temp_mat = np.array([[1,0,2,0,0],[1,-4,1,3,1],[1,-2,2,1,1],[1,1,6,7,1],[1,-1,2,1,1]])
covar_mat = temp_mat @ temp_mat.T
a = multivariate_normal(mean=[0,0,0,0,0], cov=covar_mat).pdf([[0,0,0,0,0],[1,1,1,1,1]])
print(a)
'''

# a = np.random.multivariate_normal([0,0,0,0,0], covar_mat)
# print(a)
# a = np.random.randint(0, 100, 10)
# print(a)


# images, captions, alpha, BETA, Mean, Covariance = simulation_data()
# print()
# gamma = [1,2,3,4]
# a = digamma(gamma) - digamma(sum(gamma))
# print(digamma(gamma))
# print(digamma(sum(gamma)))
# print(a)
# print(np.exp(a))


# a = np.zeros((3,3))
# a[:,1] = [2,2,2]
# print(a)

# print(np.array([1,2,3,4]) * np.array([1,2,3,4]))
# Nk = [np.zeros(3)] * 5
# print(Nk)

# Nk = np.array([[1,2,3,4]])
# Nf = np.array([[1,2,3,4]])
# print(Nf @ Nf.T)
# mat1 = (np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]))
# mat2 = (np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,5]]))
# a = (mat1 - mat2)
# print(a)
# print(a **2)
# np.mean(a)
# Nk = np.arange(5)
# Nf = np.arange(5)
#
# print(Nk @ Nf)
# vec = np.mat(np.array([1,2,3,4]))
# print(matlib.repmat(vec.T, 1,5))



# K = 3
# D = 3
# N = 4
# mu = np.zeros((K, D))
# Y = np.mat(np.array([
#     [1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3],
#     [4, 4, 4],
# ]))
#
# gamma0 = np.array([
#     [0.1, 0.8, 0.1],
#     [0.2, 0.2, 0.6],
#     [0.3, 0.3, 0.4],
#     [0.4, 0.4, 0.2]
# ])
# gamma = np.mat(np.array([
#     [0.1, 0.8, 0.1],
#     [0.2, 0.2, 0.6],
#     [0.3, 0.3, 0.4],
#     [0.4, 0.4, 0.2]
# ]))
#
#
#
# # print(gamma0[:, 2].shape)
# # print(gamma[:,2].shape)
#
# for k in range(K):
#     print(Y.shape, gamma[:, k].shape)
#     mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0)
#     cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k])
#     print(cov_k)
#


#
# # Mean = np.zeros(K,D)
#
#
# Mean = gamma.T @ Y
# print(mu)
# print(Mean)
# print()

#************
# images, captions, alpha, BETA, Mean, Covariances = simulation_data(D=10, k=2, V=5, xi=3)
# alpha_est, beta_est, Mean_est, PHI, LAMBDA= M_step_Vectorization(images=images, captions=captions,k=2,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)
# # Mean_est = np.random.normal(0, 1, (10, 5))
# print(BETA)
# print(beta_est)
#************


# alpha_mse, beta_mse, Mean_mse = mmse(alpha, BETA, Mean, alpha_est, beta_est,  Mean_est)
# print(alpha_mse, beta_mse, Mean_mse)

# plt.imshow((BETA @ beta_est.T) / 1000)
# plt.colorbar()
# plt.show()

'''
one case of getting zero value from multivariate distribution
a = np.array([[ 0.01812493,  0.00896318,  0.00759664,  0.0121407 ,  0.01321472],
       [ 0.00896318,  0.04643877, -0.02641472,  0.04473668,  0.00439261],
       [ 0.00759664, -0.02641472,  0.05096189, -0.02436573,  0.00499702],
       [ 0.0121407 ,  0.04473668, -0.02436573,  0.05148774,  0.00729521],
       [ 0.01321472,  0.00439261,  0.00499702,  0.00729521,  0.01190439]])

b= np.array([[ 0.01812493,  0.00896318,  0.00759664,  0.0121407 ,  0.01321472],
       [ 0.00896318,  0.04643877, -0.02641472,  0.04473668,  0.00439261],
       [ 0.00759664, -0.02641472,  0.05096189, -0.02436573,  0.00499702],
       [ 0.0121407 ,  0.04473668, -0.02436573,  0.05148774,  0.00729521],
       [ 0.01321472,  0.00439261,  0.00499702,  0.00729521,  0.01190439]])

am = np.array([ 1.51625027e-03, -2.27522200e-02,  1.90373597e-02, -2.27244677e-02,
   3.88784631e-03])

bm = np.array([ 7.17772002e-11, -9.86568567e-10,  8.29031276e-10, -9.86769749e-10,
   1.74394451e-10])

i = np.array([-1.71213588e+00, -3.74574326e+00, -2.52140196e-01, -1.01079516e+00,
  -4.54723256e-01])

pdf = multivariate_normal(mean=am, cov=a).pdf(i)
print(pdf)
'''



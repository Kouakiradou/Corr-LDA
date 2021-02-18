#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import multivariate_normal


def simulation_data(D=500, k=10, V=1000, xi=40, max_iter=100, gamma_shape=2, gamma_scale=1):
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
    Covariance = np.random.normal(1, 1, (k, 5))

    # image level
    N = np.random.poisson(lam=(xi - 1), size=D) + 1  # number of regions in a image, avoid 0 words
    M = np.random.poisson(lam=(xi - 1), size=D) + 1  # number of words of captions in the image
    THETA = np.random.dirichlet(alpha, D)

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N[d])
        Y = np.random.randint(0, N[d], M[d])
        temp_mean = Z @ Mean
        temp_covar = Z @ Covariance
        R = np.zeros((N[d], 5))
        W = np.zeros((M[d], V))
        for n in range(N[d]):
            covar_mat = np.diag(temp_covar[n,])
            R[n,] = np.random.multivariate_normal(temp_mean[n,], covar_mat)
        images.append(R)
        for m in range(M[d]):
            temp_BETA = Z[Y[m]] @ BETA  # we can actually manipulate the matrix in one step, fix it in the future
            W[m,] = np.random.multinomial(1, temp_BETA)
        captions.append(W)
    return images, captions, alpha, BETA, Mean, Covariance


def E_step_Vectorization(alpha, BETA, Mean, Covariance, image, caption, Phi0, gamma0, Lambda0, max_iter=100, tol=1e-3):
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

        multivari_pdf = np.zeros((image.shape[0], Mean.shape[0]))  # not sure if I should use pdf, since it is contineous
        for k in range(Mean.shape[0]):
            temp_pdf = multivariate_normal(mean=Mean[k], cov=np.diag(Covariance[k,])).pdf(image)  # Nd*1
            multivari_pdf[:, k] = temp_pdf

        ##update Phi
        Phi = multivari_pdf * np.exp(digamma(gamma) - digamma(sum(gamma))) * np.exp(Lambdaa.T @ (caption @ BETA.T)) #please double check
        Phi = Phi / (Phi.sum(axis=1)[:, None])  # row sum to 1

        ##update Lambda
        Lambdaa = np.exp((caption @ BETA.T) @ Phi.T)
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

        if (phi_delta <= tol) and (gamma_delta <= tol) and (Lambdaa_delta <= tol):
            break

    return Phi, gamma, Lambdaa


def M_step_Vectorization(images, captions, k, tol=1e-3, tol_estep=1e-3, max_iter=100, initial_alpha_shape=100,
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


    # initialization
    BETA0 = np.random.dirichlet(np.ones(V), k)
    alpha0 = np.random.gamma(shape=initial_alpha_shape, scale=initial_alpha_scale, size=k)
    Mean0 = np.random.normal(0, 1, (k, 5))
    Covariance0 = np.random.normal(1, 1, (k, 5))
    PHI = [np.ones((N[d], k)) / k for d in range(D)]
    LAMBDA = [np.ones((M[d], N[d])) / N[d] for d in range(D)]
    GAMMA = np.array([alpha0 + N[d] / k for d in range(D)]) #?? why use N[d] but bot M[d]

    BETA = BETA0
    alpha = alpha0
    alpha_dis = 1
    beta_dis = 1

    tol = tol ** 2

    for iteration in range(max_iter):

        # update PHI,GAMMA,BETA
        BETA = np.zeros((k, V))
        for d in range(D):  # documents
            PHI[d], GAMMA[d,], LAMBDA[d] = E_step_Vectorization(alpha0, BETA0, Mean0, Covariance0, images[d], captions[d], PHI[d], GAMMA[d,], LAMBDA[d], max_iter, tol_estep)
            BETA += (LAMBDA[d] @ PHI[d]).T @ captions[d]
        BETA = BETA / (BETA.sum(axis=1)[:, None])  # rowsum=1

        # update alpha

        z = D * polygamma(1, sum(alpha0))
        h = -D * polygamma(1, alpha0)
        g = D * (digamma(sum(alpha0)) - digamma(alpha0)) + (digamma(GAMMA) - digamma(GAMMA.sum(axis=1))[:, None]).sum(
            axis=0)
        c = (sum(g / h)) / (1 / z + sum(1 / h))
        alpha = alpha0 - (g - c) / h

        alpha_dis = np.mean((alpha - alpha0) ** 2)
        beta_dis = np.mean((BETA - BETA0) ** 2)
        alpha0 = alpha
        BETA0 = BETA
        if ((alpha_dis <= tol) and (beta_dis <= tol)):
            break

    return alpha, BETA


def mmse(alpha, BETA, alpha_est, BETA_est):
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
    return alpha_mse, beta_mse

# docs, alpha, BETA = simulation_data2()
# alpha_est, beta_est = M_step_Vectorization(docs=docs,k=10,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)
# alpha_mse, beta_mse = mmse(alpha, BETA, alpha_est, beta_est)
# print(alpha_mse, beta_mse)


# covar_mat = np.diag([1,1,1,1,1])
# a = multivariate_normal(mean=[0,0,0,0,0], cov=covar_mat).pdf([[0,0,0,0,0],[1,1,1,1,1]])
# print(a)


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

# print(np.array([1,2,3]) * np.array([1,2,3,4]))
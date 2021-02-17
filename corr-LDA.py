#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import digamma, polygamma


def simulation_data2(D=500, k=10, V=1000, xi=40, max_iter=100, gamma_shape=2, gamma_scale=1):
    """Simulation the data according to LDA process. Return a list
    of N_d*V matrix(one-hot-coding) with length M.
    --------------------------------------------------------------
    Input:
    number of documents M,
    number of topics k,
    number of vocabulary V,
    the parameter xi for possion distribution xi (generate the length of each document),
    the parameter gamma_shape,gamma_scale for gamma distribution (generate the alpha in the paper);
    ---------------------------------------------------------------
    Output:
    docs: documents. a length M list of Nd*V matrix (one-hot-coding),
    alpha: concentrate parameters for dirichlet distribution, k*1 vector,
    BETA: k*V matrix.

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
            R[n, ] = np.random.multivariate_normal(temp_mean[n,], covar_mat)
        images.append(R)
        for m in range(M[d]):
            temp_BETA = Z[Y[m]] @ BETA  # we can actually manipulate the matrix in one step, fix it in the future
            W[m,] = np.random.multinomial(1, temp_BETA)
        captions.append(W)
    return images, captions, alpha, BETA, Mean, Covariance


def E_step_Vectorization(alpha, BETA, doc, Phi0, gamma0, max_iter=100, tol=1e-3):
    """
    Vectorization Version Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    BETA as a k*V matrix;
    doc as a Nd*V matrix;
    Phi0 as a Nd*k matrix;
    gamma0 as a k*1 vector;
    tol as a float: tolerance.
    -------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma."""

    # Initialization
    Phi = Phi0
    gamma = gamma0
    phi_delta = 1
    gamma_delta = 1

    # relative tolerance is for each element in the matrix
    tol = tol ** 2

    for iteration in range(max_iter):
        ##update Phi
        Phi = (doc @ BETA.T) * np.exp(digamma(gamma) - digamma(sum(gamma)))
        Phi = Phi / (Phi.sum(axis=1)[:, None])  # row sum to 1

        ##update gamma
        gamma = alpha + Phi.sum(axis=0)

        ##check the convergence
        phi_delta = np.mean((Phi - Phi0) ** 2)
        gamma_delta = np.mean((gamma - gamma0) ** 2)

        ##refill
        Phi0 = Phi
        gamma0 = gamma

        if ((phi_delta <= tol) and (gamma_delta <= tol)):
            break

    return Phi, gamma


def M_step_Vectorization(docs, k, tol=1e-3, tol_estep=1e-3, max_iter=100, initial_alpha_shape=100,
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
    M = len(docs)
    V = docs[1].shape[1]
    N = [doc.shape[0] for doc in docs]

    # initialization
    BETA0 = np.random.dirichlet(np.ones(V), k)
    alpha0 = np.random.gamma(shape=initial_alpha_shape, scale=initial_alpha_scale, size=k)
    PHI = [np.ones((N[d], k)) / k for d in range(M)]
    GAMMA = np.array([alpha0 + N[d] / k for d in range(M)])

    BETA = BETA0
    alpha = alpha0
    alpha_dis = 1
    beta_dis = 1

    tol = tol ** 2

    for iteration in range(max_iter):

        # update PHI,GAMMA,BETA
        BETA = np.zeros((k, V))
        for d in range(M):  # documents
            PHI[d], GAMMA[d,] = E_step_Vectorization(alpha0, BETA0, docs[d], PHI[d], GAMMA[d,], max_iter, tol_estep)
            BETA += PHI[d].T @ docs[d]
        BETA = BETA / (BETA.sum(axis=1)[:, None])  # rowsum=1

        # update alpha

        z = M * polygamma(1, sum(alpha0))
        h = -M * polygamma(1, alpha0)
        g = M * (digamma(sum(alpha0)) - digamma(alpha0)) + (digamma(GAMMA) - digamma(GAMMA.sum(axis=1))[:, None]).sum(
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
# a = np.random.multivariate_normal([0,0,0,0,0], covar_mat)
# print(a)
# a = np.random.randint(0, 100, 10)
# print(a)


# images, captions, alpha, BETA, Mean, Covariance = simulation_data2()
# print()
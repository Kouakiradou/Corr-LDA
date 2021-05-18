import numpy as np
import joblib
from scipy.special import digamma
from scipy.stats import multivariate_normal
import torch
import imageio
from CUB_200_2011.data_processing_annotation import AnnotationProcesser


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

        multivari_pdf = np.zeros((image.shape[0], Mean.shape[0]))
        for k in range(Mean.shape[0]):
            temp_pdf = multivariate_normal(mean=Mean[k], cov=Covariances[k]).pdf(image)  # Nd*1
            multivari_pdf[:, k] = temp_pdf

        ##update Phi

        Phi = multivari_pdf * np.exp(digamma(gamma) - digamma(sum(gamma))) * np.exp(Lambdaa.T @ np.log(caption @ BETA.T)) #please double check
        dia_gamma = np.exp(digamma(gamma) - digamma(sum(gamma)))
        expppp = np.exp(Lambdaa.T @ np.log(caption @ BETA.T))
        exp = np.exp(Lambdaa.T @ np.log(caption @ BETA.T))
        diag = np.exp(digamma(gamma) - digamma(sum(gamma)))

        Phi = Phi / (Phi.sum(axis=1)[:, None])  # row sum to 1

        ##update Lambda
        Lambdaa = np.exp(np.log(caption @ BETA.T) @ Phi.T)
        Lambdaa = Lambdaa / (Lambdaa.sum(axis=1)[:, None])

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


params = np.load('/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/params/result50.npz')
scalar = joblib.load('/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/params/scaler50.pkl')

imgs_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/imgs_test_64x64.pth'
metadata_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/metadata.pth'

alpha = params['alpha']
BETA = params['BETA']
Mean = params['Mean']
Covariances = params['Covariances']

k = 50

# metadata = torch.load(metadata_path)
# print(metadata['word_id_to_word'])

image_num = 10

processor = AnnotationProcesser(imgs_path, metadata_path)
actual_img, images, captions, caption_ground_truth = processor.process(size=image_num)


index = 0
for image in actual_img:
    path = "/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/test_image/" + str(index) + ".png"
    imageio.imwrite(path, image)
    index = index + 1

for cap in caption_ground_truth:
    print(cap,'\n')
# print(caption_ground_truth)

max_iter = 100
tol_estep=1e-3
annotations = []
for i in range(len(images)):
    # annotation = []
    image = scalar.transform(images[i])
    N = images[i].shape[0]
    M = captions[i].shape[0]
    PHI0 = np.ones((N, k)) / k
    LAMBDA0 = np.ones((M, N)) / N
    GAMMA0 = np.array(alpha + N / k)
    Phi, Gamma, Lambda = E_step_Vectorization(alpha, BETA, Mean, Covariances, image, captions[i], PHI0, GAMMA0, LAMBDA0, max_iter, tol_estep)
    caption_prob = np.sum(Phi @ BETA, axis=0)
    ten_largest = np.argsort(caption_prob)[-30:]
    annotations.append([processor.dict[index] for index in ten_largest])

for annotation in annotations:
    print(annotation,'\n')

print(BETA.shape)
for i in range(k):
    ten_largest = np.argsort(BETA[i])[-30:]
    print([processor.dict[index] for index in ten_largest])

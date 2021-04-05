import imageio
import numpy as np
from colours import *
from corr_LDA import M_step_Vectorization


# result = np.zeros([400, 400, 3], np.uint8)
# result[0:200, 0:200, :] = black[0:200, 0:200, :]
# result[0:200, 200:, :] = white[0:200, 0:200, :]
# result[200:, 0:200, :] = green[0:200, 0:200, :]
# result[200:, 200:, :] = red[0:200, 0:200, :]
# imageio.imwrite("images/result.png", result)
# imageio.imwrite("images/green.png", green)

def simulation_data(D=10, k=7, gamma_shape=2, gamma_scale=1):
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

    # image level
    N = 4  # number of regions in a image
    M = np.random.randint(1, 5, D)
    THETA = np.random.dirichlet(alpha, D)

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N)
        Y = np.random.randint(0, N, M[d])
        indexes = np.arange(k) @ Z.T
        image = np.zeros([400, 400, 3], np.uint8)
        caption = [color_word[indexes[i]] for i in Y]
        Allcaption = [color_word[i] for i in indexes]
        # print(indexes)
        # print(Y)
        # print(Allcaption)
        captions.append(caption)
        image[0:200, 0:200, :] = np.clip(color[indexes[0]] + np.random.randn(3) * color_std, 0, 1) * 255
        image[0:200, 200:, :] = np.clip(color[indexes[1]] + np.random.randn(3) * color_std, 0, 1) * 255
        image[200:, 0:200, :] = np.clip(color[indexes[2]] + np.random.randn(3) * color_std, 0, 1) * 255
        image[200:, 200:, :] = np.clip(color[indexes[3]] + np.random.randn(3) * color_std, 0, 1) * 255
        images.append(image)
    return images, captions

def simulation_data2(D=10, k=7, gamma_shape=2, gamma_scale=1):
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

    # image level
    N = 4  # number of regions in a image
    M = np.random.randint(1, 5, D)
    THETA = np.random.dirichlet(alpha, D)

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N)
        Y = np.random.randint(0, N, M[d])
        indexes = np.arange(k) @ Z.T
        image = [color[i] + np.random.randn(3) * color_std for i in indexes]
        caption = [Z[i] for i in Y]
        captions.append(np.array(caption))
        images.append(np.array(image))
    return images, captions


def simulation_data3(D=10, k=7, gamma_shape=2, gamma_scale=1):

    images = []
    captions = []

    # hyperparameter
    alpha = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=k)

    # image level
    N = 4  # number of regions in a image
    M = np.random.randint(1, 5, D)
    THETA = np.random.dirichlet(alpha, D)

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N)
        Y = np.random.randint(0, N, M[d])
        indexes = np.arange(k) @ Z.T
        image = [color[i] + np.random.randn(3) * color_std for i in indexes]
        caption = [Z[i] for i in Y]
        captions.append(np.array(caption))
        images.append(np.array(image))
    return images, captions


images, captions = simulation_data2(D=500) #all topics are identical
# print("the images:-------------")
# print(images)
# print("the captions-----------")
# print(captions)
alpha_est, beta_est, Mean_est, phi, lambdaa = M_step_Vectorization(images=images, captions=captions,k=7,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)
print("beta matrix------------")
print(beta_est)
print("phi--------------------")
print(phi)
print("lambda-----------------")
print(lambdaa)
print("mean-----------------")
print(Mean_est)



# index = 0
# for image in images:
#     path = "images/" + str(index) + ".png"
#     imageio.imwrite(path, image)
#     index = index + 1
# for i in captions:
#     print(i)
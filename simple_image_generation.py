import imageio
import numpy as np
from colours import *


# result = np.zeros([400, 400, 3], np.uint8)
# result[0:200, 0:200, :] = black[0:200, 0:200, :]
# result[0:200, 200:, :] = white[0:200, 0:200, :]
# result[200:, 0:200, :] = green[0:200, 0:200, :]
# result[200:, 200:, :] = red[0:200, 0:200, :]
# imageio.imwrite("images/result.png", result)
# imageio.imwrite("images/green.png", green)

def simulation_data(D=10, k=6, gamma_shape=2, gamma_scale=1):
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
    THETA = np.random.dirichlet(alpha, D)

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N)
        indexes = np.arange(k) @ Z.T
        image = np.zeros([400, 400, 3], np.uint8)
        caption = [color_word[i] for i in indexes]
        captions.append(caption)
        image[0:200, 0:200, :] = color[indexes[0]][0:200, 0:200, :]
        image[0:200, 200:, :] = color[indexes[1]][0:200, 0:200, :]
        image[200:, 0:200, :] = color[indexes[2]][0:200, 0:200, :]
        image[200:, 200:, :] = color[indexes[3]][0:200, 0:200, :]
        images.append(image)
    return images, captions


images, captions = simulation_data()
index = 0
for image in images:
    path = "images/" + str(index) + ".png"
    imageio.imwrite(path, image)
    index = index + 1
for i in captions:
    print(i)
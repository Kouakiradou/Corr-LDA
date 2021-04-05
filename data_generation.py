import numpy as np
import colours
from colours import color_word, color_std, color
import imageio
import matplotlib.pyplot as plt
# %matplotlib inline

def make_synthetic(D = 10):
    warm = np.array([.3, .05, .09, .06, .35, .01, .01, .23])
    cold = np.array([.03, .35, .35, .2, .03, .01, .01, .02])

    TOPICS = np.stack([warm, cold])

    # hyperparameter
    K = 2
    alpha = 0.1*np.ones(K) # np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=k)

    # image level
    N = 4  # number of regions in a image
    M = np.random.randint(1, 5, D)
    THETA = np.random.dirichlet(alpha, D)


    images = []
    captions = []

    for d in range(D):
        Z = np.random.multinomial(1, THETA[d,], N)
        per_square_prob = Z @ TOPICS
        per_square = np.array([np.random.multinomial(1, per_square_prob[n]) for n in range(N)])
        im_colors = per_square @ np.array(colours.color)
        image = np.zeros([400, 400, 3], np.uint8)
        image[0:200, 0:200, :] = np.clip(im_colors[0] + np.random.randn(3) * color_std, 0, 1) * 255
        image[0:200, 200:, :] = np.clip(im_colors[1] + np.random.randn(3) * color_std, 0, 1) * 255
        image[200:, 0:200, :] = np.clip(im_colors[2] + np.random.randn(3) * color_std, 0, 1) * 255
        image[200:, 200:, :] = np.clip(im_colors[3] + np.random.randn(3) * color_std, 0, 1) * 255
        images.append(image)

        Y = np.random.permutation(N)[:M[d]]
        caption = [color_word[i] for i in np.unique(per_square[Y].argmax(-1))]
        captions.append(caption)
    return images, captions

# images, captions = make_synthetic(100)
# index = 0
# for image in images:
#     path = "images/" + str(index) + ".png"
#     imageio.imwrite(path, image)
#     index = index + 1
# for i in captions:
#     print(i)

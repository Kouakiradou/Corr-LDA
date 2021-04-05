import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np

from skimage import io
# from scipy import io

from skimage.data import astronaut, chelsea
from skimage import measure
import gensim
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


# labelPath = "dataset/imagelabels.mat"
# label = io.loadmat(labelPath)
# setidPath = "dataset/setid.mat"
# setid = io.loadmat(setidPath)
# print(label)
# print(imlist)

def DataTrans(x, length):
    """Turn the data into the desired structure"""

    N_d = len(x)
    V = length

    row = 0

    doc = np.zeros((N_d, V))
    for i in range(len(x)):
        doc[i, x[i][0]] = x[i][1]
        row += 1

    return doc


def image_process(origin_images):
    images = []
    for img in origin_images:
        segments_slic = slic(img, n_segments=4, compactness=10, sigma=1, start_label=1)
        N = len(np.unique(segments_slic))
        regions = measure.regionprops(segments_slic, intensity_image=img)
        image = np.zeros((N, 3))
        for i in range(len(regions)):
            image[i] = regions[i].mean_intensity
        images.append(image)
    return images


def caption_process(origin_captions):
    dictionary = gensim.corpora.Dictionary(origin_captions)
    dicLength = dictionary.__len__()
    bow_corpus = [dictionary.doc2bow(caption) for caption in origin_captions]
    captions = [DataTrans(d, dicLength) for d in bow_corpus]
    return captions, dictionary

# image_process()

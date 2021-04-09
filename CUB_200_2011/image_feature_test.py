import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import mahotas.features.texture as texture
from skimage import io
import cv2

from skimage.data import astronaut, chelsea
from skimage import measure
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte

img = img_as_float(io.imread("/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/images/0.png"))
segments_slic = slic(img, n_segments=30, compactness=13, sigma=1,
                     start_label=1)
print(segments_slic)
c = np.unique(segments_slic)
regions = measure.regionprops(segments_slic, intensity_image=img)
print(len(regions))
if 0 in c:
	c = np.delete(c, 0)
for (i, segVal) in enumerate(c):
	# construct a mask for the segment
	print(i)
	new = img_as_ubyte(img)
	new[new == 0] = 1
	gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
	# print(i)
	# print(new)
	mask = np.zeros(gray.shape, dtype = np.bool)
	mask[segments_slic != segVal] = True
	gray[mask] = 0
	features = texture.haralick(gray, ignore_zeros=True)
	ht_mean = features.mean(axis=0)
	a = np.hstack((regions[i].mean_intensity,regions[i].centroid, ht_mean))
	# print(regions[i].mean_intensity)
	# print(regions[i].centroid)
	# print(ht_mean)
	# print(a.shape)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)


ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
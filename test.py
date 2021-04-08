from scipy.stats import multivariate_normal
import numpy as np
import torch
import imageio
import torchvision
from skimage.util import img_as_float
from skimage import io

#one case of multivariate distribution
# temp_mat = np.array([[1,0,2,0,0],[1,-4,1,3,1],[1,-2,2,1,1],[1,1,6,7,1],[1,-1,2,1,1]])
# covar_mat = temp_mat @ temp_mat.T
# a = multivariate_normal(mean=[0,0,0,0,0], cov=covar_mat).pdf([[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2]])
# print(a)
#
# multivari_pdf = np.zeros((3, 5))
#
# multivari_pdf[:, 0] = a
# print(multivari_pdf)


# a = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
# b = np.array([2,2,2,2])
# c = a * b
# print(c)
# d = np.mat(c[:, 1]).T
# print(d)
# print(np.multiply(c, d))

# url = "dataset/archive/metadata.pth"
# meta = torch.load(url)
# print(meta.keys())
# print(meta['train_val_img_ids'])
# print(meta['img_ids'])
# print(meta['img_id_to_encoded_caps'][18])
# print(meta['class_name_to_class_id'])
# print(len(meta['word_id_to_word']))
# print(meta['num_words'])
# print(meta['word_to_word_id'])
# print(meta['num_captions_per_image'])

url2 = 'dataset/archive/imgs_train_val_128x128.pth'
train_val = torch.load(url2)
print(train_val[2].shape)
# pilimage = torchvision.transforms.ToPILImage()(train_val[0])
new = train_val[2].permute(1, 2, 0)
print(img_as_float(new))

path = "images/0.png"
imageio.imwrite(path, new)



# image0url = 'dataset/archive/cub_200_2011_64x64_for_fid_10k/cub_200_2011_64x64_10k/0.png'
# image8854url = 'dataset/archive/cub_200_2011_64x64_for_fid_10k/cub_200_2011_64x64_10k/8854.png'
#
# img = io.imread(image0url)
# print(img)

# def caption_to_one_hot(caption):
#     one_hot_caption = np.zeros((len(caption), 15))
#     for i in range(len(caption)):
#         one_hot_caption[i, caption[i]] = 1
#     print(one_hot_caption)
#     return one_hot_caption
#
# caption_to_one_hot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))





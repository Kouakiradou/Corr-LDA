from CUB_200_2011.data_processing import Processer
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_spd_matrix
imgs_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/imgs_train_val_64x64.pth'
metadata_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/metadata.pth'
result_parameter = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/result.npz'

processed_data_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/processed_data.npz'
# captionsPath = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/'

processor = Processer(imgs_path, metadata_path)
images, captions = processor.process(size=10)
#
# # w = np.empty((0,19))
# # c = np.ones((1,19))
# scalar = preprocessing.StandardScaler()
# scalar.fit(np.vstack(images))
# new_image = [scalar.transform(image) for image in images]
np.savez(processed_data_path, images=images, captions=captions)
# # for

# b = np.ones((3,3))
# print(b)
# a = np.identity(b.shape[0]) * 0.01
# print(a+b)
# c = make_spd_matrix(3)
# print(c)




# print(images[0])
# scalar = preprocessing.StandardScaler()
# images_data = [scalar.fit_transform(X) for X in images]
# # print(images)
# temp = np.asarray(images)
# print(temp.shape)
# new_img = temp.reshape(-1, temp.shape[-1])
# # print(new_img)
# print
# print(scalar.mean_)
# # print(images, [caption.shape for caption in captions])
# np.savez(processed_data_path, images=images, captions=captions)

# data = np.load(result_parameter, allow_pickle=True)
# print(data['LAMBDA'])

# Covariances = [np.zeros((5, 5)) for i in range(3)]
# print(Covariances)
# Covariances[0][0][0] = 1
# print(Covariances)

# mat = [np.zeros((3,3)) for _ in range(5)]
# mat[0] += np.ones((3,3))
# print(mat)
# data = np.load(processed_data_path, allow_pickle=True)
# new_img = data['images']
# new_cap = data['captions']
# print(new_img[0])
# print([image.shape for image in new_img])
# print([caption.shape for caption in new_cap])
# print(type(new_img))
# print(type(new_cap))
# print(type(new_img[0]))


# scalar = preprocessing.StandardScaler()
# new_images = np.asarray(images)
# print(type(images))
# print(type(new_images))
# a = new_images.reshape(-1, new_images.shape[-1])
# print(a)
# scalar.fit(new_images.reshape(-1, new_images.shape[-1]))
# scaled_images = [scalar.transform(img) for img in images]
# print(scaled_images)


# for i in range(k):
#     Covariances[i] += np.identity(dim) * 0.001
#
# Covariances0.append(make_spd_matrix(dim))
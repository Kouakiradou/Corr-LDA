import numpy as np
from scipy.stats import multivariate_normal


# data = np.load('/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/mean_cova.npz')
# mean = data['mean']
# cov = data['cov']
# img = data['img']
# print(mean)
# print(cov)
# print(img)
# temp_pdf = multivariate_normal(mean=mean, cov=cov).pdf(img)
# print(temp_pdf)

# para_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/result.npz'
# para = np.load(para_path, allow_pickle=True)
# phi = para['PHI']
# print(phi)

mat = np.array([[1,1,1],[2,2,2],[3,3,3]])
mat2 = np.array([[2,2,2],[3,3,3],[4,4,4]])
list = np.asarray([mat,mat2])
print(mat ** 2)
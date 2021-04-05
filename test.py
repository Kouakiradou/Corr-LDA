from scipy.stats import multivariate_normal
import numpy as np


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


a = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
b = np.array([2,2,2,2])
c = a * b
print(c)
d = np.mat(c[:, 1]).T
print(d)
print(np.multiply(c, d))
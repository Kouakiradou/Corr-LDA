import numpy as np
from scipy.stats import multivariate_normal


data = np.load('/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/mean_cova.npz')
mean = data['mean']
cov = data['cov']
img = data['img']
print(mean)
print(cov)
print(img)
temp_pdf = multivariate_normal(mean=mean, cov=cov).pdf(img)
print(temp_pdf)
from corr_LDA import M_step_Vectorization
import numpy as np

path = ''
data = np.load(path, allow_pickle=True)

images = data['images']
captions = data['captions']

alpha, BETA, Mean, Covariances, PHI, LAMBDA, GAMMA = M_step_Vectorization(images=images, captions=captions,k=10,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)

result_parameter = ''
np.savez(result_parameter, alpha=alpha, BETA=BETA, Mean=Mean, Covariances=Covariances, PHI=PHI, LAMBDA=LAMBDA, GAMMA=GAMMA)
from corr_LDA import M_step_Vectorization
import numpy as np
from sklearn import preprocessing
import joblib


print("loading...")
path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/processed_data.npz'
data = np.load(path, allow_pickle=True)

images = data['images']
captions = data['captions']

print("scaling...")
scaler = preprocessing.MinMaxScaler()
scaler.fit(np.vstack(images))
new_image = [scaler.transform(image) for image in images]
joblib.dump(scaler, "scaler.pkl")

print("start training")
alpha, BETA, Mean, Covariances, PHI, LAMBDA, GAMMA = M_step_Vectorization(images=new_image, captions=captions,k=30,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)

result_parameter = 'result.npz'
np.savez(result_parameter, alpha=alpha, BETA=BETA, Mean=Mean, Covariances=Covariances, PHI=PHI, LAMBDA=LAMBDA, GAMMA=GAMMA)
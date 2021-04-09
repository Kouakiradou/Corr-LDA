from CUB_200_2011.data_processing import Processer
import numpy as np
imgs_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/imgs_train_val_64x64.pth'
metadata_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/metadata.pth'


processed_data_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/processed_data.npz'
# captionsPath = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/CUB_200_2011/preprocessed_data/'

processor = Processer(imgs_path, metadata_path)
images, captions = processor.process()
# print(images, [caption.shape for caption in captions])
np.savez(processed_data_path, images=images, captions=captions)

data = np.load(processed_data_path, allow_pickle=True)
new_img = data['images']
new_cap = data['captions']
print([image.shape for image in new_img])
print([caption.shape for caption in new_cap])
print(type(new_img))
print(type(new_cap))
print(type(new_img[0]))
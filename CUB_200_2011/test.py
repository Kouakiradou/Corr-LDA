from CUB_200_2011.data_processing import Processer

imgs_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/imgs_train_val_64x64.pth'
metadata_path = '/Users/kouakiradou/Machine Learning/Final-Year-project/Corr-LDA/dataset/archive/metadata.pth'

processor = Processer(imgs_path, metadata_path)
images, captions = processor.process()
print(images, [caption.shape for caption in captions])
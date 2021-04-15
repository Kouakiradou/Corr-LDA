import torch
import numpy as np
import cv2
from skimage import measure
from skimage.util import img_as_float, img_as_ubyte
from skimage.segmentation import slic
import mahotas.features.texture as texture

imgs_path = 'dataset/archive/imgs_train_val_64x64.pth'
metadata_path = 'dataset/archive/metadata.pth'


class Processer:
    def __init__(self, data_path, meta_path):
        metadata = torch.load(meta_path)
        self.img_ids = metadata['train_val_img_ids']
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.num_words = metadata['num_words']
        self.num_captions_per_image = metadata['num_captions_per_image']
        self.imgs = torch.load(data_path)

    def process(self, size=None):
        captions = []
        images = []
        if size is None:
            size = len(self.imgs)
        for idx in range(size):
            print(idx)
            img = self.imgs[idx]
            img_id = self.img_ids[idx]

            # processing image to properties of segmentation
            images.append(self.img_feature_extraction(img))

            # processing captions to on-hot-coding
            encoded_caps = self.img_id_to_encoded_caps[img_id]
            cap_idx = torch.randint(low=0, high=10, size=(1,)).item()
            encoded_cap = encoded_caps[cap_idx]
            captions.append(self.caption_to_one_hot(encoded_cap))
        return images, captions

    def img_segmentation(self, img):
        # Change CHW to HWC
        HWC_img = img_as_float(img.permute(1, 2, 0))
        segments_slic = slic(HWC_img, n_segments=10, compactness=10, sigma=1, start_label=1)
        N = len(np.unique(segments_slic))
        regions = measure.regionprops(segments_slic, intensity_image=HWC_img)
        image = np.zeros((N, 19))
        for i in range(len(regions)):
            image[i] = regions[i].mean_intensity
        return image

    def img_feature_extraction(self, img):
        # Change CHW to HWC
        HWC_img = img_as_float(img.permute(1, 2, 0))
        segments_slic = slic(HWC_img, n_segments=20, compactness=10, sigma=1, start_label=1)
        unique_lables = np.unique(segments_slic)
        if 0 in unique_lables:
            unique_lables = np.delete(unique_lables, 0)
        N = len(unique_lables)
        image = np.zeros((N, 19))
        # image = np.zeros((N, 6))
        regions = measure.regionprops(segments_slic, intensity_image=HWC_img)
        for (i, segVal) in enumerate(unique_lables):
            # convert image to grayscale
            ubyte_img = img_as_ubyte(HWC_img)
            ubyte_img[ubyte_img == 0] = 1
            gray = cv2.cvtColor(ubyte_img, cv2.COLOR_BGR2GRAY)

            # construct a mask for the segment
            mask = np.zeros(gray.shape, dtype=np.bool)
            mask[segments_slic != segVal] = True
            gray[mask] = 0

            # extract texture feature from mahotas package
            texture_features = texture.haralick(gray, ignore_zeros=True)
            ht_mean = texture_features.mean(axis=0)

            # extract mean_intensity feature
            mean_intensity = regions[i].mean_intensity

            # extract centroid feature
            centroid = regions[i].centroid

            #extract area feature
            area = regions[i].area

            image[i] = np.hstack((mean_intensity,centroid, area, ht_mean))
            # image[i] = np.hstack((mean_intensity, centroid, area))
        return image

    def caption_to_one_hot(self, caption):
        one_hot_caption = np.zeros((len(caption), self.num_words))
        for i in range(len(caption)):
            one_hot_caption[i, caption[i]] = 1
        return one_hot_caption



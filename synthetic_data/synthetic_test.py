from synthetic_data.data_generation import make_synthetic
from synthetic_data.data_processing import image_process, caption_process
import imageio
import heapq
from skimage.util import img_as_float
from skimage import io
from corr_LDA import M_step_Vectorization

print("generating synthetic data...")
origin_images, origin_captions = make_synthetic(500)  # origin_captions be like ['Green','BLUE', 'YELLOW']


index = 0
for image in origin_images:
    path = "images/" + str(index) + ".png"
    imageio.imwrite(path, image)
    index = index + 1

print("reading data from file...")
image_from_file = []
for i in range(500):
    img = img_as_float(io.imread(f"images/{i}.png"))
    image_from_file.append(img)

print("processing data...")
images = image_process(image_from_file)
captions, dictionary = caption_process(origin_captions)
print(dictionary.token2id)

print("training...")
alpha_est, beta_est, Mean_est, phi, lambdaa = M_step_Vectorization(images=images, captions=captions,k=2,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)
print("beta matrix------------")
print(beta_est)
# print("phi--------------------")
# print(phi)
# print("lambda-----------------")
# print(lambdaa)
print("mean-----------------")
print(Mean_est)


def find_index(x):
    """find the index of the largest 10 values in a list"""

    x = x.tolist()
    max_values = heapq.nlargest(10, x)
    index = [0] * 8
    for i in range(8):
        index[i] = x.index(max_values[i])

    return index

rep_words_index = list(map(find_index, beta_est))
print([dictionary[i] for i in rep_words_index[0]])
print([dictionary[i] for i in rep_words_index[1]])

# for i in captions:
#     print(i)

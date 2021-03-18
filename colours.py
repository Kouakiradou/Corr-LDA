import numpy as np

# red = np.zeros([400, 400, 3], np.uint8)
# red[:, :, 0] = np.zeros([400, 400]) + 255
#
# green = np.zeros([400, 400, 3], np.uint8)
# green[:, :, 1] = np.zeros([400, 400]) + 255
#
# blue = np.zeros([400, 400, 3], np.uint8)
# blue[:, :, 2] = np.zeros([400, 400]) + 255
#
# black = np.zeros([400, 400, 3], np.uint8)
#
# white = np.zeros([400, 400, 3], np.uint8)
# white[:, :, 0] = np.zeros([400, 400]) + 255
# white[:, :, 1] = np.zeros([400, 400]) + 255
# white[:, :, 2] = np.zeros([400, 400]) + 255

from PIL import Image
color = []
color_word = []
color.append(np.asarray(Image.new("RGB", (400,400), "green")))
color_word.append("green")

color.append(np.asarray(Image.new("RGB", (400,400), "blue")))
color_word.append("blue")

color.append(np.asarray(Image.new("RGB", (400,400), "red")))
color_word.append("red")

color.append(np.asarray(Image.new("RGB", (400,400), "black")))
color_word.append("black")

color.append(np.asarray(Image.new("RGB", (400,400), "white")))
color_word.append("white")

color.append(np.asarray(Image.new("RGB", (400,400), "yellow")))
color_word.append("yellow")
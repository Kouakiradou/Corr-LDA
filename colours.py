import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

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

# from PIL import Image
# color = []
# color_word = []
# color.append(np.asarray(Image.new("RGB", (400,400), "green")))
# color_word.append("green")
#
# color.append(np.asarray(Image.new("RGB", (400,400), "blue")))
# color_word.append("blue")
#
# color.append(np.asarray(Image.new("RGB", (400,400), "red")))
# color_word.append("red")
#
# color.append(np.asarray(Image.new("RGB", (400,400), "black")))
# color_word.append("black")
#
# color.append(np.asarray(Image.new("RGB", (400,400), "white")))
# color_word.append("white")
#
# color.append(np.asarray(Image.new("RGB", (400,400), "yellow")))
# color_word.append("yellow")


# image = np.zeros((400, 400, 3), np.uint8)
# image[0:200, 0:200] = np.array([1,0,0]) * 255
# imageio.imwrite("images/test.png", image)
# print(image)
# green = [[1]]

# Mean = np.zeros((5,3)) #rgb
# Mean[0] = np.array([0.95,0,0])  #red
# Mean[1] = np.array([0,0.95,0])  #green
# Mean[2] = np.array([0,0,0.95])  #blue
# Mean[3] = np.array([0.95,0.95,0.95])  #white
# Mean[4] = np.array([0,0,0])  #black
# print(Mean)


RED = np.array([0.9, 0.1, 0.1])
GREEN = np.array([0.1, 0.8, 0.1])
BLUE = np.array([0.1, 0.1, 0.9])
CYAN = np.array([0.1, 0.8, 0.9])
ORANGE = np.array([0.9, 0.6, 0.1])
WHITE = np.array([0.9, 0.9, 0.9])
BLACK = np.array([0.1, 0.1, 0.1])
YELLOW = np.array([0.9, 0.9, 0.1])

color_std = np.array([0.04, 0.04, 0.04])

color = [RED, GREEN, BLUE, CYAN, ORANGE, BLACK, WHITE, YELLOW]
color_word = ["RED", "GREEN", "BLUE", "CYAN", "ORANGE", "BLACK", "WHITE", "YELLOW"]
import numpy as np
from PIL import Image


def load_image(filename: str) -> np.array:
    img = Image.open(filename)
    return np.asarray(img)


def convert_to_gray(
        input_image: np.array) -> np.array:
    height, width, depth = input_image.shape
    result_image: np.array = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            result_image[y][x] \
                = input_image[y][x][0] * 0.299 \
                + input_image[y][x][1] * 0.587 \
                + input_image[y][x][2] * 0.114
    return result_image


def convert_to_binary(
        input_image: np.array,
        threshold: int = 127) -> np.array:
    max_val: int = 255
    min_val: int = 0
    initial_conv: np.array = np.where((input_image <= threshold), input_image, max_val)
    final_conv: np.array = np.where((initial_conv > threshold), initial_conv, min_val)
    return final_conv


def invert_image(
        input_image: np.array) -> np.array:
    height, width = input_image.shape
    result_image: np.array = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            result_image[y][x] = 255 - input_image[y][x]
    return result_image

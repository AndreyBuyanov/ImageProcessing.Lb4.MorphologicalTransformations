import numpy as np
from typing import Callable


def dilation_comparator(
        input_image_segment: np.array,
        kernel: np.array) -> bool:
    height, width = input_image_segment.shape
    for x in range(width):
        for y in range(height):
            if kernel[y][x] == 0:
                continue
            if kernel[y][x] == input_image_segment[y][x]:
                return True
    return False


def erosion_comparator(
        input_image_segment: np.array,
        kernel: np.array) -> bool:
    height, width = input_image_segment.shape
    for x in range(width):
        for y in range(height):
            if kernel[y][x] != input_image_segment[y][x]:
                return False
    return True


def abstract_kernel(
        input_image: np.array,
        kernel: np.array,
        image_x: int,
        image_y: int,
        comparator: Callable[[np.array, np.array], bool]) -> bool:
    kernel_height, kernel_width = kernel.shape
    kernel_width_half: int = kernel_width // 2
    kernel_height_half: int = kernel_height // 2
    x_pos_begin = image_x - kernel_width_half
    x_pos_end = image_x + kernel_width_half + 1
    y_pos_begin = image_y - kernel_height_half
    y_pos_end = image_y + kernel_height_half + 1
    input_image_segment: np.array = input_image[y_pos_begin:y_pos_end, x_pos_begin:x_pos_end]
    return comparator(input_image_segment, kernel)


def abstract_operation(
        input_image: np.array,
        kernel: np.array,
        comparator: Callable[[np.array, np.array, int, int, int, int], bool]) -> np.array:
    kernel_height, kernel_width = kernel.shape
    padding_width: int = kernel_width // 2
    padding_height: int = kernel_height // 2
    padding = ((padding_height, padding_height), (padding_width, padding_width))
    input_image_padding: np.array = np.pad(
        array=input_image,
        pad_width=padding,
        mode='constant',
        constant_values=0)
    result_image: np.array = np.zeros(input_image.shape)
    image_height, image_width = result_image.shape
    for image_x in range(image_width):
        for image_y in range(image_height):
            if (abstract_kernel(
                    input_image=input_image_padding,
                    kernel=kernel,
                    image_x=image_x+padding_width,
                    image_y=image_y+padding_height,
                    comparator=comparator)):
                result_image[image_y][image_x] = 255
    return result_image


def dilation(
        input_image: np.array,
        kernel: np.array) -> np.array:
    return abstract_operation(
        input_image=input_image,
        kernel=kernel,
        comparator=dilation_comparator)


def erosion(
        input_image: np.array,
        kernel: np.array) -> np.array:
    return abstract_operation(
        input_image=input_image,
        kernel=kernel,
        comparator=erosion_comparator)


def opening(
        input_image: np.array,
        kernel: np.array) -> np.array:
    erosion_image: np.array = erosion(
        input_image=input_image,
        kernel=kernel)
    dilation_image: np.array = dilation(
        input_image=erosion_image,
        kernel=kernel)
    return dilation_image


def closing(
        input_image: np.array,
        kernel: np.array) -> np.array:
    dilation_image: np.array = dilation(
        input_image=input_image,
        kernel=kernel)
    erosion_image: np.array = erosion(
        input_image=dilation_image,
        kernel=kernel)
    return erosion_image

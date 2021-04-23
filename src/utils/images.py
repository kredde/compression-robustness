from PIL import Image
import numpy as np
import math
from typing import *
from pathlib import Path

MIN_REQ_SIZE = 32


def load_image_as_numpy(image_path: Union[str, Path]) -> Tuple[np.array, Tuple[int, int]]:
    """Loads an image and represents it as numpy array. If it is smaller than 32x32, it will be upscaled

    :param image_path: File path to the image
    :return: The loaded image represented as a int32 numpy array and the size of the image previous to any potential
        rescaling
    """
    image = Image.open(image_path)

    original_size = (image.width, image.height)

    # Resize the image if it is too small for the functions in the imagecorruptions package
    if original_size[0] < MIN_REQ_SIZE and original_size[0] <= original_size[1]:
        image = image.resize((MIN_REQ_SIZE, math.ceil(
            MIN_REQ_SIZE / original_size[0] * original_size[1])))
    elif original_size[1] < MIN_REQ_SIZE:
        image = image.resize(
            (math.ceil(MIN_REQ_SIZE / original_size[1] * original_size[0]), MIN_REQ_SIZE))

    return np.asarray(image, dtype='uint8'), original_size


def save_numpy_as_image(image_array: np.array, output_path: Union[str, Path],
                        rescale_to: Optional[Tuple[int, int]] = None, greyscale: bool = False) -> None:
    """Saves a numpy array as an image to disk

    :param image_array: The image represented as a numpy array
    :param output_path: The file path where the image will be saved to
    :param rescale_to: If provided, rescales to the specified size
    :param greyscale: True, if the image should be saved as greyscale image
    :return: None
    """
    image = Image.fromarray(np.asarray(
        np.clip(image_array, 0, 255), 'uint8'), 'L' if greyscale else 'RGB')
    if rescale_to is not None:
        image = image.resize(rescale_to)
    image.save(output_path)

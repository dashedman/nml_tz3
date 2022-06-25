import numpy as np
from enum import Enum

import selectivesearch


class LumCoef:
    RED = 0.299
    GREEN = 0.587
    BLUE = 0.114

def gray_scale(image_data: np.array):
    gray_image_data = np.zeros((image_data.shape[:2]), 'uint')
    for i, line in enumerate(image_data):
        for j, pixel in enumerate(line):
            gray_pixel = int(
                        LumCoef.RED * pixel[0] +
                        LumCoef.GREEN * pixel[1] +
                        LumCoef.BLUE * pixel[2]
                    )
            gray_image_data[i][j] = gray_pixel
    return gray_image_data

def seek_objects(image_data: np.array):
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    yield from regions

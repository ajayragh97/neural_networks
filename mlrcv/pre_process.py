import numpy as np
import torch
from typing import Optional

def heatmap_object(img: np.ndarray, bounding_box: dict, heatmap: np.ndarray) -> np.ndarray:
    """
    This function generates the heatmaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - heatmap (np.ndarray): heatmap of the current input img

    Returns:
        - heatmap (np.ndarray): output heatmap with the current object heatmap added
    """
    extend = bounding_box['bndbox']
    center = bounding_box['center']
    x0 = center[0]
    y0 = center[1]
    xmax = extend['xmax']
    xmin = extend['xmin']
    ymax = extend['ymax']
    ymin = extend['ymin']
    sigma = 5
    A = 1
    
    for i in range(xmin, xmax, 1):
        for j in range(ymin, ymax, 1):
            x = ((i - x0)**2) / (2 * sigma**2)
            y = ((j - y0)**2) / (2 * sigma**2)
            heatmap[i,j] = A * np.exp(-(x + y))
    return heatmap

def sizemap_object(img: np.ndarray, bounding_box: dict, sizemap: np.ndarray) -> np.ndarray:
    """
    This function generates the sizemaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - sizemap (np.ndarray): sizemap of the current input img

    Returns:
        - sizemap (np.ndarray): output sizemap with the current object sizemap added
    """
    
    extend = bounding_box['bndbox']
    center = bounding_box['center']
    
    x0 = center[0]
    y0 = center[1]
    xmax = extend['xmax']
    xmin = extend['xmin']
    ymax = extend['ymax']
    ymin = extend['ymin']

    height = ymax - ymin
    width = xmax - xmin
    sizemap[x0, y0, 0] = height
    sizemap[x0, y0, 1] = width

    return sizemap
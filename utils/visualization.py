"""Shared BEV visualization helpers (class palette + colorizer)."""

from typing import List

import numpy as np

# Six BEV classes, in label-index order.
CLASS_NAMES: List[str] = [
    'background',
    'drivable_area',
    'lane_divider',
    'road_divider',
    'walkway',
    'ped_crossing',
]

# RGB colors matching the paper's figures, in label-index order.
CLASS_COLORS = np.array([
    [252, 252, 252],  # background
    [166, 206, 227],  # drivable area
    [202, 178, 214],  # lane divider
    [106, 61, 154],   # road divider
    [224, 74, 76],    # walkway
    [251, 154, 153],  # ped crossing
], dtype=np.uint8)


def colorize_bev(class_map: np.ndarray) -> np.ndarray:
    """Map a [H, W] array of class indices to an [H, W, 3] uint8 RGB image."""
    class_map = np.asarray(class_map)
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        rgb[class_map == idx] = color
    return rgb

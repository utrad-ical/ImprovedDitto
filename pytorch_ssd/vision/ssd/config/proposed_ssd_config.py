"""
reference
[1]:
[2]:
[3]:
"""

import numpy as np
import sys

sys.path.append("....")
sys.path.append("../pytorch_ssd/")

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 256
image_mean = np.array([127, 127, 127]) # RGB layout
image_std = 1.0

iou_threshold = 0.2
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(128, 2, SSDBoxSizes(5, 15), [2]),
    SSDSpec(64, 4, SSDBoxSizes(15, 30), [2, 3]),
    SSDSpec(32, 8, SSDBoxSizes(30, 50), [2, 3]),
    SSDSpec(16, 16, SSDBoxSizes(50, 80), [2, 3]),
    SSDSpec(8, 32, SSDBoxSizes(80, 100), [2]),
    SSDSpec(4, 64, SSDBoxSizes(100, 130), [2])
]

priors = generate_ssd_priors(specs, image_size)


class ProposedSSDConfig():
    def __init__(self):
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        
        self.iou_threshold = iou_threshold
        self.center_variance = center_variance
        self.size_variance = size_variance
        
        self.specs = specs
        
        self.priors = priors

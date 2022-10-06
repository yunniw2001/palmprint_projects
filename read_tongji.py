import random

import numpy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import mytransform


def prepare_transform_for_image():
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((224, 224))
    contrast_brightness_adjustment = transforms.ColorJitter(brightness=50, contrast=0.5)
    smooth_or_sharpening = transforms.Compose([
        mytransform.MeanFiltersTransform(),
        mytransform.MedianFiltersTransform(),
        mytransform.GaussFiltersTransform(),
        mytransform.GaussianFiltersTransformUnsharpMask(),
        mytransform.MedianFiltersTransformUnsharpMask(),
        mytransform.MeanFiltersTransformUnsharpMask()
    ])
    color_shift = transforms.ColorJitter(hue=0.14)
    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.RandomApply(rotation, 0.6),
                transforms.RandomApply(resized_cropping, 0.6),
                transforms.RandomApply(contrast_brightness_adjustment, 0.6),
                transforms.RandomApply(transforms.RandomChoice(smooth_or_sharpening), 0.6),
                transforms.RandomApply(color_shift, 0.6),
                transforms.Normalize(128, 128)
            ]
        )
    }

import random

import numpy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2


class MeanFiltersTransform:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            tmp_img = cv2.blur(tmp_img, (1,1))
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class MeanFiltersTransformUnsharpMask:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            blur_img = cv2.blur(tmp_img, (1,1))
            tmp_img = tmp_img-blur_img+tmp_img
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img

class MedianFiltersTransform:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            tmp_img = cv2.medianBlur(tmp_img, 3)
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class MedianFiltersTransformUnsharpMask:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            blur_img = cv2.medianBlur(tmp_img, 3)
            tmp_img = tmp_img-blur_img+tmp_img
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class GaussFiltersTransform:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            tmp_img = cv2.GaussianBlur(tmp_img, (5,5),sigmaX=0,sigmaY=0)
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class GaussianFiltersTransformUnsharpMask:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            blur_img = cv2.GaussianBlur(tmp_img, (5,5),sigmaX=0,sigmaY=0)
            tmp_img = tmp_img-blur_img+tmp_img
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


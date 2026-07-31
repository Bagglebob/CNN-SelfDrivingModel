import cv2
import numpy as np


def crop(img):
    """Keep the road region (matches TestSimulation.py)."""
    return img[60:135, :, :]


def to_yuv(img):
    """RGB -> YUV."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def blur(img, ksize=(3, 3)):
    """Gaussian blur."""
    return cv2.GaussianBlur(img, ksize, 0)


def resize(img, size=(200, 66)):
    """Nvidia model input size (width, height)."""
    return cv2.resize(img, size)


def normalize(img):
    """Scale pixels to [0, 1]."""
    return img / 255.0


def preprocess(img):
    """
    Full preprocess pipeline.
    Expects RGB image (convert with cv2.COLOR_BGR2RGB after imread).
    Order matches TestSimulation.preProcessing.
    """
    img = crop(img)
    img = to_yuv(img)
    img = blur(img)
    img = resize(img)
    img = normalize(img)
    return img
import cv2
import numpy as np


def normalize_image(image: np.ndarray, size=(48, 48)):  # pragma: no cover
    image = cv2.resize(image, size)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)

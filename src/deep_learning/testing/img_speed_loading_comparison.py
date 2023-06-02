import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_pil(path):
    return np.asarray(Image.open(path)).transpose((2, 0, 1))


def load_image_opencv(path: str, channel_first: bool = True, resize_hw: tuple = None):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, resize_hw[::-1]) if resize_hw is not None else img
    return img.transpose((2, 0, 1)) if channel_first else img


if __name__ == "__main__":
    for i in tqdm(range(1000)):
        load_image_opencv("custom_data/test.png")
        # load_image_pil("custom_data/test.png")

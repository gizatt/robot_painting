'''
    Expanded from output of ChatGPT 4o on 20240526
'''
import numpy as np
import cv2
import os
import random
from PIL import Image

class BackgroundImageLoader:
    '''
        Maintains three classes of images, and provides them as random backgrounds with some probability:
        1) Blank white image
        2) Random crops of a supplied dataset of images
        3) User-supplied additional runtime-augmented images. (E.g. images that have had paintstrokes added to them.)
        Some percent of the time, returns a blank white image.
        Some percent of the time, returns a random crop of a random image in the dataset.
    '''

    BLANK_PROBABILITY = 0.3
    AUGMENTED_IMAGE_PROBABILTIY = 0.3

    def __init__(self, directory):
        self.directory = directory
        self.dataset_images = []
        self.augmented_images = []
        self.load_images()

    def add_supplied_image(self, image):
        self.augmented_images.append(Image.fromarray(image))

    def load_images(self):
        """Loads all images from the specified directory into memory."""
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(root, file)
                    try:
                        with Image.open(img_path) as img:
                            self.dataset_images.append(img.copy())
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        assert len(self.dataset_images) > 0

    def get_random_background_from_dataset(self, img_width, img_height, images) -> np.ndarray:
        """Returns a random background image cropped to the specified width and height."""
        assert len(images) > 0
        num_attempts = 0
        while num_attempts <= 100:
            background = random.choice(images)
            bg_width, bg_height = background.size
            
            if bg_width < img_width or bg_height < img_height:
                continue
            
            if bg_width == img_width:
                left = 0
            else:
                left = np.random.randint(0, bg_width - img_width)
            if bg_height == img_height:
                top = 0
            else:
                top = np.random.randint(0, bg_height - img_height)
            right = left + img_width
            bottom = top + img_height
            
            return np.array(background.crop((left, top, right, bottom))).copy()

        raise ValueError(f"With {num_attempts} found no images that fit the image requirements")

    def get_random_background(self, img_width, img_height) -> np.ndarray:
        """Returns a random background image cropped to the specified width and height."""
        select = np.random.random()
        if select <= self.BLANK_PROBABILITY:
            return np.ones((img_width, img_height, 3))
        elif select <= self.BLANK_PROBABILITY + self.AUGMENTED_IMAGE_PROBABILTIY and len(self.augmented_images) > 0:
            return self.get_random_background_from_dataset(img_width, img_height, self.augmented_images)
        else:
            return self.get_random_background_from_dataset(img_width, img_height, self.dataset_images)


if __name__ == "__main__":
    loader = BackgroundImageLoader("C:/Users/Greg Izatt/Pictures/ffxiv")

    for k in range(100):
        cropped_background = loader.get_random_background(128, 128)
        cv2.imshow("random crop", cropped_background)
        cv2.waitKey(30)
import cv2
import numpy as np
import random
import albumentations as A
from skimage.util import random_noise

import cv2
import numpy as np
import random
import albumentations as A
from skimage.util import random_noise

class NoiseGenerator:
    def __init__(self, image, seed=None):
        self.image = image.copy()
        self.seed = seed
        
        # Set the global seeds if provided upon initialization
        if self.seed is not None:
            self.set_seed(self.seed)
            
        self.is_grayscale = len(self.image.shape) == 2
        if self.is_grayscale:
            self.h, self.w = self.image.shape
        else:
            self.h, self.w = self.image.shape[:2]

    def set_seed(self, seed):
        """Sets the seed for all randomness sources to ensure reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        # Albumentations relies on Python's random and NumPy's random, 
        # so setting them above covers the Albumentations transformations too.

    def _get_organic_mask(self):
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        margin_w = max(1, int(self.w * 0.2))
        margin_h = max(1, int(self.h * 0.2))
        
        center_x = random.randint(margin_w, self.w - margin_w)
        center_y = random.randint(margin_h, self.h - margin_h)
        
        max_radius = max(3, int(min(self.w, self.h) * 0.3))
        min_radius = max(2, int(max_radius * 0.5))
        base_radius = random.randint(min_radius, max_radius)
        
        cv2.circle(mask, (center_x, center_y), base_radius, 255, -1)
        
        for _ in range(3):
            offset_x = random.randint(-base_radius, base_radius)
            offset_y = random.randint(-base_radius, base_radius)
            radius = random.randint(max(1, base_radius // 2), base_radius)
            cv2.circle(mask, (center_x + offset_x, center_y + offset_y), radius, 255, -1)
            
        blur_k = max(3, int(min(self.w, self.h) * 0.3))
        if blur_k % 2 == 0:
            blur_k += 1 
            
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        mask = mask.astype(float) / 255.0
        
        if self.is_grayscale:
            return mask 
        else:
            return np.dstack([mask, mask, mask]) 

    def _blend(self, modified_image, mask):
        orig_f = self.image.astype(float)
        mod_f = modified_image.astype(float)
        blended = (orig_f * (1.0 - mask)) + (mod_f * mask)
        return blended.astype(np.uint8)

    def gen_blur_noise(self, kernel_size=9): 
        mask = self._get_organic_mask()
        blurred = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return self._blend(blurred, mask)

    def gen_mblur_noise(self, kernel_size=9):
        mask = self._get_organic_mask()
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
        motion_blurred = cv2.filter2D(self.image, -1, kernel)
        return self._blend(motion_blurred, mask)

    def gen_speckle_noise(self, var=0.08):
        mask = self._get_organic_mask()
        img_normalized = self.image / 255.0
        # Pass the seed directly to scikit-image's random_noise
        speckled = random_noise(img_normalized, mode='speckle', var=var, rng=self.seed)
        speckled = np.array(255 * speckled, dtype=np.uint8)
        return self._blend(speckled, mask)

    def gen_stain_noise(self, intensity=0.7):
        mask = self._get_organic_mask()
        if self.is_grayscale:
            stain_layer = np.full_like(self.image, 50, dtype=np.uint8) 
        else:
            stain_layer = np.full_like(self.image, (30, 60, 100), dtype=np.uint8) 
        stain_effect = cv2.addWeighted(self.image, 1 - intensity, stain_layer, intensity, 0)
        return self._blend(stain_effect, mask)

    def gen_smudge_noise(self, opacity=0.5, thickness=2, color=255):
        """
        Simulates a smudge or finger swipe as a soft white stroke/line passing through the image.
        """
        stroke_mask = np.zeros_like(self.image, dtype=np.uint8)
        
        # Pick start, middle, and end points to form a natural curved line
        start_pt = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))
        mid_pt = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))
        end_pt = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))
        
        # Format stroke color for grayscale vs RGB
        stroke_color = color if self.is_grayscale else (color, color, color)
        
        # Draw curved stroke
        pts = np.array([start_pt, mid_pt, end_pt], np.int32).reshape((-1, 1, 2))
        cv2.polylines(stroke_mask, [pts], isClosed=False, color=stroke_color, thickness=thickness)
        
        # Apply a light blur so the stroke has soft "smudged" edges
        blur_k = 3 if min(self.w, self.h) <= 64 else 5
        stroke_mask = cv2.GaussianBlur(stroke_mask, (blur_k, blur_k), 0)
        
        # Blend the stroke over the image
        return cv2.addWeighted(self.image, 1.0, stroke_mask, opacity, 0)

    def gen_intensity_diff_noise(self, gamma=2.5):
        mask = self._get_organic_mask()
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        intensity_adjusted = cv2.LUT(self.image, table)
        return self._blend(intensity_adjusted, mask)


def generate_3_noisy_variations(image):
    generator = NoiseGenerator(image)
    
    # List of all available method names
    noise_methods = [
        'gen_blur_noise',
        'gen_mblur_noise',
        # 'gen_speckle_noise',
        'gen_stain_noise',
        'gen_smudge_noise',
        'gen_intensity_diff_noise'
    ]
    
    # Randomly select 3 unique method names
    chosen_names = random.sample(noise_methods, k=3)
    
    # Generate images by calling each method dynamically using getattr
    noisy_images = [getattr(generator, name)() for name in chosen_names]
    
    # Returns ( [img1, img2, img3], ['method1', 'method2', 'method3'] )
    chosen_names = [name.split('_')[1] for name in chosen_names]
    return noisy_images, chosen_names

if __name__ == "__main__":
    from framework.utils.plot_all import plot_all
    import os
    from glob import glob
    import cv2 as cv

    from tqdm import tqdm

    path = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_CORE_CLEAN"

    # ---- #2 for dataset generation
    out_path = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_CORE_NOISE"
    for file in tqdm(glob(os.path.join(path, '*'))):
        img = cv.imread(file, 0)
        # NG = NoiseGenerator(img, None)
        # # Blur
        # blur_img = NG.gen_blur_noise(kernel_size=9)
        # motion_blur_img = NG.gen_motion_blur_noise(kernel_size=9)
        # plot_all([img, blur_img, motion_blur_img], title_list=["Original", "Blur", "Motion_Blur"], normalization=True)
        # # Speckle
        # speckle_img = NG.gen_speckle_noise(var=0.08)
        # # Stain
        # stain_img = NG.gen_stain_noise(intensity=0.7)
        # # Smudge
        # smudge_img = NG.gen_smudge_noise(opacity=0.5, thickness=2, color=255)
        # # Intensity difference
        # gamma_img = NG.gen_intensity_diff_noise(gamma=2.5)
        # plot_all([img, speckle_img, stain_img, smudge_img, gamma_img], title_list=["Original", "Speckle", "Stain", "Smudge", "Gamma"], normalization=True)
        # # plot_all([])

        # ---- Generate Dataset
        from framework.utils.fourier import Fourier2D
        img_list, name_list = generate_3_noisy_variations(img)
        for i in range(3):
            name = os.path.basename(file).split('.')[0]+f"_{name_list[i]}.png"
            cv.imwrite(os.path.join(out_path, name), img_list[i])
        # plot_all(img_list, title_list=name_list)
        
        # ---- ---- ----

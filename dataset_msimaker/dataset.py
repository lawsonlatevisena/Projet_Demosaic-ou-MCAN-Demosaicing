from os import listdir
from os.path import join
import torch.utils.data as data
from tifffile import TiffFile  # Changed from libtiff
from PIL import Image
import numpy as np
import random
from My_function import reorder_imec, mask_input

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def load_img(filepath):
    # Replace TIFFfile with TiffFile
    with TiffFile(filepath) as tif:
        img = tif.asarray()  # This returns a numpy array
        if img.ndim == 3:  # If already in (height, width, channels)
            img = img.transpose(2, 1, 0)  # Adjust dimensions if needed
        return img

def randcrop(a, crop_size):
    [wid, hei, nband] = a.shape
    crop_size1 = crop_size
    Width = random.randint(0, wid - crop_size1 - 1)
    Height = random.randint(0, hei - crop_size1 - 1)
    return a[Width:(Width + crop_size1), Height:(Height + crop_size1), :]

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, norm_flag, input_transform=None, target_transform=None, augment=False):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        random.shuffle(self.image_filenames)
        self.crop_size = calculate_valid_crop_size(128, 4)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        self.norm_flag = norm_flag

    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        input_image = input_image.astype(np.float32)
        
        if self.norm_flag:
            norm_name = 'maxnorm'
            max_raw = np.max(input_image)
            max_subband = np.max(np.max(input_image, axis=0), 0)
            norm_factor = max_raw / max_subband
            for bn in range(16):
                input_image[:, :, bn] = input_image[:, :, bn] * norm_factor[bn]
                
        input_image = randcrop(input_image, self.crop_size)
        
        if self.augment:
            if np.random.uniform() < 0.5:
                input_image = np.fliplr(input_image)
            if np.random.uniform() < 0.5:
                input_image = np.flipud(input_image)
            input_image = np.rot90(input_image, k=np.random.randint(0, 4))
            
        target = input_image.copy()
        input_image = mask_input(target, 4)
        input_image = reorder_imec(input_image)
        target = reorder_imec(target)
        
        if self.input_transform:
            raw = input_image.sum(axis=2)
            raw = self.input_transform(raw)/255.0
            input_image = self.input_transform(input_image)/255.0
            
        if self.target_transform:
            target = self.target_transform(target)/255.0

        return raw, input_image, target

    def __len__(self):
        return len(self.image_filenames)
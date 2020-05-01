from sklearn.datasets import load_sample_images
from sklearn.feature_extraction import image
from PIL import Image
import glob
import numpy as np
import cv2
import os

image_dir=glob.glob('test_images_greyscale/*.jpg')

patches_dir = 'patches/'

dataset= np.array([np.array(Image.open(fname)) for fname in image_dir])

i=800

for images in range(0,1000):
    patches = image.extract_patches_2d(dataset[images],(256,256),200,15)
    
    
    cv2.imwrite(os.path.join(patches_dir,str(i)+'.jpg'),patches[images])
    
    i=i+1


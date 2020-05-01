import numpy as np
import cv2
import os
import glob
from PIL import Image

image_dir=glob.glob('test_patches/*.jpg')



dataset= np.array([np.array(Image.open(fname)) for fname in image_dir])

#os.mkdir('test_noise_patches')
noise_patches_dir = 'test_noise_patches/'

noise_factor=0.4
mean = 0
var = 0.5
sigma = var**0.5

row,col,ch= dataset.shape

gauss = np.random.normal(mean,sigma,(row,col,ch))

gauss = gauss.reshape(row,col,ch)

noise_images = dataset + gauss

print(np.sum(dataset[100]))
print(np.sum(noise_images[100])) 


'''
#noise_images = dataset + noise_factor * np.random.normal(loc=0,scale=1.0,size=dataset.shape)



i=0
for images in range(0,200):
    
    
    cv2.imwrite(os.path.join(noise_patches_dir,str(i)+'.jpg'),noise_images[images])
    i+=1 

'''

import cv2
import glob, os, errno

# Replace mydir with the directory you want
img_dir= glob.glob(os.path.join('opensource_images/','*.'+'png'))
save_dir= 'opensource_images_greyscale/'

for fil in glob.glob(img_dir):
    image = cv2.imread(fil) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    cv2.imwrite(os.path.join(save_dir,fil),gray_image)

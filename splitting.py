import os
import shutil
import glob
import sys

source_dir = glob.glob(os.path.join('all_images_patches/','*.'+'jpg'))

print(len(source_dir))

img_dir = 'all_images_patches/'

dest_dir_train = 'train_patches/'

dest_dir_test = 'test_patches/'

i=0

for aindex,aimages in enumerate(source_dir):

    base_name_images = (aimages.split('/')[-1]).split('.')[0]
    
    source_label = img_dir + base_name_images + '.jpg'
    

    shutil.move(source_label,dest_dir_test)
    
    i = i+1
    
    if i > 799:
        sys.exit()  


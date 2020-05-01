import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob 
import argparse
import os

####Add model directory as an argument, Please make sure it is empty####
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', action='store', dest='model_dir', help='Directory for saving logs and checkpoints')
args = parser.parse_args()
root_dir= "/mnt/data2/test_optimus/"

tf.enable_eager_execution()

model = models.Sequential()

model.add(layers.Conv2D(18, (2, 2), activation='relu', input_shape=(256, 256, 1), padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding = 'same'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding = 'same'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding = 'same'))
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding = 'same'))

model.add(layers.MaxPooling2D((1, 1), padding='same'))

model.add(layers.Conv2D(16, (2, 2), activation='relu', padding = 'same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(16, (1, 1), activation='relu'))
#model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'))
#model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'))
model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))

model.summary()

train_data_dir=glob.glob('train_patches/*.jpg')
test_data_dir=glob.glob('test_patches/*.jpg')

x_train= np.array([np.array(Image.open(fname)) for fname in train_data_dir])
x_test= np.array([np.array(Image.open(fname)) for fname in test_data_dir])

noise_factor=25
mean = 0
var = 0.5
sigma = var**0.5

row,col,ch= x_train.shape
row1,col1,ch1= x_test.shape

gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss1= np.random.normal(mean,sigma,(row1,col1,ch1))
gauss = gauss.reshape(row,col,ch,1)
gauss1 = gauss1.reshape(row1,col1,ch1,1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = tf.reshape(x_train, (len(x_train), 256, 256,1)) #Clear Train Images (output)
y_test = tf.reshape(x_test, (len(x_test), 256, 256,1)) #Clear Test Images (output)

####Adding Guassian Noise####
x_train_noisy = y_train + gauss 
x_test_noisy = y_test + gauss1

x_train = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.) #Noisy Train Images (input)
x_test = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.) #Noisy Test Images (input)

model.compile(optimizer='adam',
          loss='mean_squared_error',
          metrics=['accuracy'])
          
####Please make sure this exsists####          
checkpoint_path = root_dir + args.model_dir + "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

####Creating a callback for saving checkpoints####
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=800)
    
####Creating a callback for early stopping####
'''earlystop_callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_acc', min_delta=0.0001,
  patience=10)'''

####Saving the checkpoints after every 100 epochs####    
model.save_weights(checkpoint_path.format(epoch=0))

####Please make sure this exsists####          
log_dir = os.path.join((root_dir + args.model_dir + "/logs/fit/"), (datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

####Creating a callback for logging progress to tensorboard####
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

          
history = model.fit(x_train, y_train, batch_size=100, epochs=100, 
                    validation_data=(x_test, y_test), callbacks = [tensorboard_callback, cp_callback])

####For Evaluations from Saved checkpoints####
latest = tf.train.latest_checkpoint(checkpoint_dir)                    
model.load_weights(latest)
loss,acc = model.evaluate(x_test,  y_test, verbose=2)
print("Final Evaluation:\nAccuracy: {:5.2f}%".format(100*acc))                    
                                        
####FOR OUTPUTING A SAMPLE OF 10 IMAGES####                    
predicted_images = model.predict(x_train)

n = 10
plt.figure(figsize=(20, 4)) 
for i in range(n):

    # display original    
    ax = plt.subplot(2, n, i + 1)
    x_train = np.squeeze(x_train)    
    plt.imshow(x_train[i])    
    plt.gray()    
    ax.get_xaxis().set_visible(False)    
    ax.get_yaxis().set_visible(False)
    plt.title('Input Noisy Image')

    # display reconstruction    
    ax = plt.subplot(2, n, i+1+n)  
    reconstruction = np.squeeze(predicted_images)    
    plt.imshow(reconstruction[i])    
    plt.gray()    
    ax.get_xaxis().set_visible(False)    
    ax.get_yaxis().set_visible(False)
    plt.title('Output De-Noised Image') 
    plt.show()




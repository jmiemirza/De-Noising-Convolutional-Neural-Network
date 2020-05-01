import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import matplotlib.pyplot as plt
import glob 
import numpy as np

tf.enable_eager_execution()

model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding = 'same'))

model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(8, (3, 3), activation='relu', padding = 'same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.summary()

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

noise_factor=5
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
y_train = tf.reshape(x_train, (len(x_train), 28, 28,1)) #Clear Train Images (output)
y_test = tf.reshape(x_test, (len(x_test), 28, 28,1)) #Clear Test Images (output)

###Adding Guassian Noise###
x_train_noisy = y_train + gauss 
x_test_noisy = y_test + gauss1

x_train = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.) #Noisy Train Images (input)
x_test = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.) #Noisy Test Images (input)

model.compile(optimizer='adam',
          loss='mean_squared_error',
          metrics=['accuracy'])
          
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
          
history = model.fit(x_train, y_train, batch_size=60, epochs=1000, 
                    validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
                    
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






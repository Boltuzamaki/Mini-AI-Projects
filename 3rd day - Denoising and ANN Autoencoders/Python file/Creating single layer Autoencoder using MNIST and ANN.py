#!/usr/bin/env python
# coding: utf-8

# #  Creating single layer Autoencoder using MNIST and ANN

# In[1]:


from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Preprocessing/ reshaping input images

(X_train, _), (X_test, _) = mnist.load_data()                        # Loading data from mnist

X_train = X_train.astype('float32')/255                              # normalizing the data for fast training
X_test = X_test.astype('float32')/255

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))  # Reshape to flat from 2-D image array     
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

print(X_train.shape)
print(X_test.shape)


# In[3]:


input_img= Input(shape=(784,))     # Converting to input image into keras tensor
input_img.shape


# In[4]:


encoded = Dense(units = 32, activation = 'relu')(input_img)              # Encoder layer


# In[5]:


decoded = Dense(units= 784, activation = 'sigmoid')(encoded)             # Decoder layer


# In[6]:


autoencoder = Model(input_img, decoded)                     # Creating autoencoder which is made of encoder and decoder


# In[7]:


autoencoder.summary()  


# In[8]:


encoder = Model(input_img, encoded)                       # Creating autoencoder only


# In[9]:


encoder.summary()


# In[10]:


encoded_input = Input(shape=(32,))                        # All image is compressed into size 32 latent vector


# In[11]:


decoder_layer = autoencoder.layers[-1]                    # Creting decoder layer 


# In[12]:


# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


# In[13]:


decoder.summary()


# In[14]:


# Comlpiling and fitting model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy',metrics = ['accuracy'])

autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 256, shuffle = True, validation_data = (X_test, X_test))


# In[15]:


# Saving model and weights of autoencoder
model_json = autoencoder.to_json()
with open("simple_autoencoder_model", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("simple_autoencoder_model.h5"),
print("Saved model to disk")


# In[16]:


# Saving model and weights of encoder
model_json = encoder.to_json()
with open("simple_autoencoder_model_encoder_only", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder.save_weights("simple_autoencoder_model_encoder_only.h5"),
print("Saved model to disk")


# In[17]:


# Saving model and weights of decoder
model_json = decoder.to_json()
with open("simple_autoencoder_model_decoder_only", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
decoder.save_weights("simple_autoencoder_model_decoder_only.h5"),
print("Saved model to disk")


# In[18]:


encoded_imgs = encoder.predict(X_test)                         # Passing image through encoder to create latent vector
predicted = autoencoder.predict(X_test)                        # Passing image through whole autoencoder
decoder_predict = decoder.predict(encoded_imgs)                # Pass the encoded value by encoder via decoder to check if it can generate mnist like image or not


# In[19]:


# Plotting figure 

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded image
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs[i].reshape(8,4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(decoder_predict[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
plt.show()

# First one is real image
# Second one is encoded latent vector
# Thir one is the image which decoder generated from encoded value


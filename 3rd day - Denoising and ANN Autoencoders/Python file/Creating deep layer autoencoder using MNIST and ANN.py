#!/usr/bin/env python
# coding: utf-8

# # Creating deep layer autoencoder using MNIST and ANN

# In[1]:


from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Preprocessing/ reshaping input images

(X_train, _), (X_test, _) = mnist.load_data()              # Load mnist data

X_train = X_train.astype('float32')/255                    # Normalize mnist data
X_test = X_test.astype('float32')/255

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:])) # Reshaping image to 1-D
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

print(X_train.shape)
print(X_test.shape)


# In[3]:


input_img= Input(shape=(784,))     # Converting to input image into keras tensor
input_img.shape


# In[4]:


layer1 = Dense(units = 128, activation = 'relu')(input_img)        # encoder layers
layer2 = Dense(units = 64, activation = 'relu')(layer1)
layer3 =  Dense(units = 32, activation = 'relu')(layer2)        # We find Encodings here
layer4 = Dense(units = 64, activation = 'relu')(layer3)          # Decoder layers
layer5 = Dense(units = 128, activation = 'relu')(layer4)
final_layer =Dense(units = 784, activation = 'sigmoid')(layer5)   ## decoded image


# In[5]:


autoencoder = Model(input_img, final_layer)      # Creating autoencoder model


# In[6]:


autoencoder.summary()


# In[7]:


encoder = Model(input_img, layer3)                     # Creating encoder model


# In[8]:


encoder.summary()


# In[9]:


# Creating decoder separately

encoded_input = Input(shape=(32,))
decoder_layer3 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer3)
decoder_layer1 = autoencoder.layers[-1](decoder_layer2) 


# In[10]:


decoder = Model(encoded_input, decoder_layer1)


# In[11]:


decoder.summary()


# In[12]:


# Compile the mdoel
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


# Train the model
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))


# In[14]:


# Saving model and weights of autoencoder
model_json = autoencoder.to_json()
with open("deep_autoencoder_model", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("deep_autoencoder_model.h5"),
print("Saved model to disk")


# In[15]:


# Saving model and weights of encoder
model_json = encoder.to_json()
with open("deep_autoencoder_model_encoder_only", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder.save_weights("deep_autoencoder_model_encoder_only.h5"),
print("Saved model to disk")


# In[16]:


# Saving model and weights of decoder
model_json = decoder.to_json()
with open("deep_autoencoder_model_decoder_only", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
decoder.save_weights("deep_autoencoder_model_decoder_only.h5"),
print("Saved model to disk")


# In[17]:


encoded_imgs = encoder.predict(X_test)                 # Here the image is converted into encoded vector
predicted = autoencoder.predict(X_test)                # The autoencoder taking real image and produce a image
decoder_predict = decoder.predict(encoded_imgs)        # The decoder producing image using encoded vector


# In[18]:


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

# The first one is real image
# Sencond one is the encoded value
# This one is the image created by encoded vector


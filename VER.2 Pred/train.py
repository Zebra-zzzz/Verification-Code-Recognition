
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
from keras import layers, optimizers, regularizers
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from keras.models import Model

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)


# In[2]:


from keras.utils.np_utils import to_categorical

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


# In[3]:


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
plt.imshow(X[0])
plt.title(decode(y))


# In[4]:


from keras.models import *
from keras.layers import *

def cnnModel(input_shape):
    x = Input(input_shape)
    X_Input = x
    for i in range(4):
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(inputs=X_Input, outputs=x, name='cnnModel')
    return model


# In[5]:


input_shape = (height, width, 3)
model = cnnModel(input_shape)


# In[6]:


from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[7]:


#model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
#                   validation_data=gen(), nb_val_samples=1280)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('zzw.h5', verbose=1, save_best_only=True)
model.fit_generator(gen(), samples_per_epoch = 25600, epochs = 10, validation_steps = 40, validation_data = gen(), callbacks=[earlystopper, checkpointer])


# In[8]:


model1 = load_model('zzw.h5')


# In[9]:


model1.summary()


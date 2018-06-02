
# coding: utf-8

# In[5]:


import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers, regularizers
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Input,
                          MaxPooling2D, ZeroPadding2D, ActivityRegularization)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils, plot_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from matplotlib.pyplot import imshow
from utils import *

import string


# In[2]:


characters = string.digits + string.ascii_uppercase
print(characters)
n_class, n_len = len(characters), 4 #一共36个字符，每个验证码4个字符

K.set_image_data_format('channels_last')


# In[3]:


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_datasets()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

Y = [np.zeros((Y_train.shape[0], n_class), dtype=np.uint8) for i in range(n_len)]
for i, y in enumerate(Y_train):
    y = y.decode()
    for j, ch in enumerate(y):
        # print(y)
        # y[j][i, :] = 0
        Y[j][i, characters.find(ch)] = 1
        # Y[i][j, characters.find(ch)] = 1
# print(Y)

Z = [np.zeros((Y_test.shape[0], n_class), dtype=np.uint8) for i in range(n_len)]
for i, y in enumerate(Y_test):
    y = y.decode()
    for j, ch in enumerate(y):
        # print(y)
        # y[j][i, :] = 0
        Z[j][i, characters.find(ch)] = 1
        # Y[i][j, characters.find(ch)] = 1
# print(Z)


# In[6]:


def CNNModel(input_shape):
    X_input = Input(input_shape)
    x = X_input
    # for i in range(4):
    #     x = ZeroPadding2D((1,1))
    #     x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    #     x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    #     x = MaxPooling2D((2, 2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(64, (7,7), activation='relu', kernel_regularizer=regularizers.l2(0.03), name='conv0')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(128, (5,5), activation='relu', kernel_regularizer=regularizers.l2(0.03), name='conv1')(x)
    x = MaxPooling2D((3,3))(x)
    x = Conv2D(256, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.03), name='conv2')(x)
    
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu', name='fc0')(x)
    # x = Dense(10000, activation='relu', name='FC0')(x)
    # x = Dense(1024, activation='relu', name='FC1')(x)

    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(inputs=X_input, outputs=x, name="CNNModel")
    return model


# In[7]:


# happyModel = HappyModel((64,64,3))
# lrs = [3e-4, 5e-4, 1e-4, 7e-5]
lrs = [3e-4]
for lr in lrs:

    cnnModel = CNNModel((60, 240, 3))
    # cnnModel.summary()
    # 需要修改loss_function
    # happyModel.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                    epsilon=None, decay=0.0, amsgrad=False)
    cnnModel.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # categorical_crossentropy

    # happyModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=25)
    cnnModel.fit(x=X_train, y=Y, epochs=3, batch_size=128)

    # preds = cnnModel.evaluate(x=X_test, y=Z)

    # print()
    # print ("Loss = " + str(preds[0]))
    # print ("Test Accuracy = " + str(preds[1]))
# preds = happyModel.evaluate(x=X_test, y=Y_test,)
cnnModel.save('model_315.h5')


# In[ ]:


from tqdm import tqdm
def evaluate(model, batch_num=30):
    batch_acc = 0
    for i in tqdm(range(batch_num)):
        X = X_test[i]
        y = []
        for z in Y:
            y.append(z[i])
        y_pred = model.predict(np.asarray([X]))
        acc = np.array_equal(np.array([np.argmax(y, axis=1).T]), np.argmax(y_pred, axis=2).T)
        #print(y, y_pred)
        ArgA = np.array([np.argmax(y, axis=1).T])
        ArgPred = np.argmax(y_pred, axis=2).T 
        print(ArgA, ArgPred)
        print(np.array_equal(ArgA, ArgPred))
        #print(np.argmax(y, axis=1), np.argmax(y_pred, axis=2))
        batch_acc += acc
    return batch_acc / (batch_num * 4)

evaluate(model)


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import imshow


# In[14]:


from PIL import Image


# In[ ]:


im = Image.open('./images/wMp6.jpg')
img = np.asarray(im)
imshow(img)
y_pred = cnnModel.predict(np.asarray([img]))


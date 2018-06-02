# Verification-Code-Recognition
## About this project
这是《用户体验设计》这门课的第一个大作业，要求实现对验证码（英文字母+数字）的识别。作业示例使用了PIL库读取图片，并利用了光学字符识别包pytesseract，依赖tesseract ORC engine进行图像识别。但最后的识别率并不是很高，仅为22.8%，所以选择了目前更为主流的深度学习进行验证码的训练和验证。

完整的项目分为两个部分，分别对应一个验证版本。对于第一个版本，尽管用于训练的数据集最终的训练结果非常可观，但是在对剩下30张用于test的图片进行验证时，正确率却出奇的低，怀疑是出现了过拟合的问题。第二个版本对神经网络结构做了一定更改，重新构造了更为复杂卷积神经网络，以减少样本量的需求，并尽量避免过拟合的问题。同时，考虑到显卡本身的性能，准备定义一个数据生成器在训练过程中利用CPU无限生成大量数据。

___

## VER.1  Verification

### 数据集收集

因为本次识别打算利用深度学习框架Keras，用Python建立一个卷积神经网络进行训练，所以需要大量的数据集。
因此选择了直接通过以下代码(`/VER.1 Verification/genPic.py`)**自动生成24000张训练图片和30张验证图片**,全部放在`/VER.1 Verification/images`的文件夹内:
 ```py
 import random
import sys
# from uuid import uuid1

import h5py
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

def rnd_char():
    '''
    随机一个字母或者数字
    :return: 
    '''
    # 随机一个字母或者数字
    i = random.randint(1,3)
    if i == 1:
        # 随机个数字的十进制ASCII码
        an = random.randint(97, 122)
    elif i == 2:
        # 随机个小写字母的十进制ASCII码
        an = random.randint(65, 90)
    else:
        # 随机个大写字母的十进制ASCII码
        an = random.randint(48, 57)
    # 根据Ascii码转成字符，return回去
    return chr(an)

#　干扰
def rnd_dis():
    '''
    随机一个干扰字
    :return: 
    '''
    d = ['^','-', '~', '_', '.']
    i = random.randint(0, len(d)-1)
    return d[i]

# 两个随机颜色都规定不同的区域，防止干扰字符和验证码字符颜色一样
# 随机颜色1:
def rnd_color():
    '''
    随机颜色，规定一定范围
    :return: 
    '''
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 随机颜色2:
def rnd_color2():
    '''
      随机颜色，规定一定范围
      :return: 
      '''
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

def create_code(code_list):
    # 240 x 60:
    width = 60 * 4
    height = 60
    image = Image.new('RGB', (width, height), (192, 192, 192))
    # 创建Font对象:
    font = ImageFont.truetype('fonts/CAMBRIA.TTC', 44)

    # 创建Draw对象:
    draw = ImageDraw.Draw(image)

    # 填充每个像素:
    for x in range(0, width, 20):
        for y in range(0, height, 10):
            draw.point((x, y), fill=rnd_color())

    # 填充字符
    _str = ""
    # 填入4个随机的数字或字母作为验证码
    for t in range(4):
        c = rnd_char()
        _str = "{}{}".format(_str, c)

        # 随机距离图片上边高度，但至少距离30像素
        h = random.randint(1, height-50)
        # 宽度的化，每个字符占图片宽度1／4,在加上10个像素空隙
        w = width/4 * t + 10
        draw.text((w, h), c, font=font, fill=rnd_color2())
        # draw.text((w, h), c, fill=rnd_color2())

    # 实际项目中，会将验证码 保存在数据库，并加上时间字段
    print("保存验证码 {} 到数据库".format(_str))

    # 给图片加上字符干扰，密集度由 w, h控制
    for j in range(0, width, 30):
        dis = rnd_dis()
        w = t * 15 + j

        # 随机距离图片上边高度，但至少距离30像素
        h = random.randint(1, height - 30)
        draw.text((w, h), dis, font=font, fill=rnd_color())
        # draw.text((w, h), dis, fill=rnd_color())

    # 模糊:
    image.filter(ImageFilter.BLUR)

    code_list.append(_str)
    # uuid1 生成唯一的字符串作为验证码图片名称
    # code_name = '{}.jpg'.format(uuid1())
    code_name = '{}.jpg'.format(_str)
    save_dir = './images/{}'.format(code_name)
    image.save(save_dir, 'jpeg')
    print("已保存图片: {}".format(save_dir))

# 当直接运行文件的是和，运行下面代码
if __name__ == "__main__":
    code_list = []
    for i in range(24000):
        create_code(code_list)
    f = h5py.File("datasets/codes.hdf5","w")
    code_list_asc = []
    for i in code_list:
        code_list_asc.append(i.encode())
    d1 = f.create_dataset('train_codename', data=code_list_asc)

    code_list = []
    for i in range(30):
        create_code(code_list)
    # f = h5py.File("datasets/codes.hdf5","w")
    code_list_asc = []
    for i in code_list:
        code_list_asc.append(i.encode())
    d1 = f.create_dataset('test_codename', data=code_list_asc)
```

同时利用以下代码（`/VER.1 Verification/createDataset.py`）**生成datesets**并放入了`/VER.1 Verification/datasets`的文件夹内：
```py
import h5py
from PIL import Image
import numpy as np

def createTrainSet():
    trainList = []
    f=h5py.File("./datasets/codes.hdf5","r+")
    dset_code = f['train_codename']
    # print(dset.shape)
    for i in dset_code.value:
        # print(i.decode())
        name = i.decode()
        img = Image.open("./images/{}.jpg".format(name))
        imgMatrix = np.array(img)
        trainList.append(imgMatrix)
    # print(trainList[1].shape)
    f.create_dataset('train_images', data=trainList)

def createTestSet():
    testList = []
    f=h5py.File("./datasets/codes.hdf5","r+")
    dset_code = f['test_codename']
    # print(dset.shape)
    for i in dset_code.value:
        # print(i.decode())
        name = i.decode()
        img = Image.open("./images/{}.jpg".format(name))
        imgMatrix = np.array(img)
        testList.append(imgMatrix)
    # print(trainList[1].shape)
    f.create_dataset('test_images', data=testList)


if __name__ == '__main__':
    createTrainSet()
    createTestSet()
```

每张图片由任意4个数字和大小写英文字母组成，并同时加入随机不同颜色的若干干扰字。

**图片示例如下**：

![验证码图片样例](https://github.com/Zebra-zzzz/Verification-Code-Recognition/blob/master/VER.1%20%20Verification/Verification-Code-sample.png)

### 方法挑选

**本实验采用Python软件作为英文文字识别的工具**

虽然Python现在已有较为常用的光学字符识别包pytesseract，但是最后的识别率并不是很高。所以选择了目前更为主流的深度学习进行验证码的训练和验证。查阅了相关资料后，CNN特别适用于对于图片的识别，故决定选用名为Keras的深度学习框架搭建CNN。在参考学习了Coursera上由deeplearning.ai开设的深度学习课程（[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)）之后，具体为第四门课（[Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)）第二周目名为The Happy House的model（用于识别人脸是否开心）（`/VER.1 Verification/Keras+-+Tutorial+-+Happy+House+v2.ipynb`），自行对应教学文档的格式写出了适用于验证码识别的程序，并修改了部分卷积神经网络的结构。

**完整代码（训练+验证）**（`/VER.1 Verification/model.py`）如下，jupyter notebook格式（`/VER.1 Verification/model.ipynb`）：
```py
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
```
**NOTE:** *选择CNN进行训练的原因*：

在一般的神经网络中，每个像素都都与每一个神经元相连。增加的计算负荷使得网络在这种情况下不太准确。

而对于CNN来说，并不是所有上下层神经元都能直接相连，而是通过“卷积核”作为中介。卷积神经网络使得图像处理可以通过邻近度对连接进行滤波而计算可管理。在给定层中，卷积神经网络不是将每个输入连接到每个神经元，而是有意限制了连接，使得任何一个神经元仅从它之前的层的小部分接受输入（例如5 * 5或3 * 3像素）。因此，每个神经元只负责处理图像的某一部分。同一个卷积核在所有图像内是共享的，图像通过卷积操作后仍然保留原先的位置关系。

对于图像，如果没有卷积操作，学习的参数量是灾难级的。CNN之所以用于图像识别，正是由于CNN模型限制了参数的个数并挖掘了局部结构的这个特点。

### 分析与评价
#### 训练结果

在第三回合训练的时候一般即可收敛，最后的Accuracy已经达到0.82：
```
Epoch 1/3
24000/24000 [==============================] - 1170s 49ms/step - loss: 12.9443 - c1_loss: 2.9272 - c2_loss: 2.9516 - c3_loss: 2.9329 - c4_loss: 2.9379 - c1_acc: 0.3486 - c2_acc: 0.3381 - c3_acc: 0.3480 - c4_acc: 0.3427
Epoch 2/3
24000/24000 [==============================] - 1339s 56ms/step - loss: 8.9131 - c1_loss: 2.1092 - c2_loss: 2.1352 - c3_loss: 2.0903 - c4_loss: 2.2616 - c1_acc: 0.4440 - c2_acc: 0.4325 - c3_acc: 0.4480 - c4_acc: 0.4156
Epoch 3/3
24000/24000 [==============================] - 1313s 55ms/step - loss: 2.4174 - c1_loss: 0.4274 - c2_loss: 0.4765 - c3_loss: 0.4352 - c4_loss: 0.5489 - c1_acc: 0.8611 - c2_acc: 0.8466 - c3_acc: 0.8535 - c4_acc: 0.8263
```
#### 验证结果

尽管用于Train的数据集最终的训练结果非常可观，但是在对剩下30张用于test的图片进行验证时，正确率却出奇的低，怀疑是出现了**过拟合的问题**：
```
100%|██████████████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00,  19.56it/s]
0.0
```

虽然试着**加大了数据集**（从1024张最终加到了24000张），正确率却没有明显的变化。在查阅了相关文档后，考虑到验证码的字符有36种（26+10），利用CNN网络进行训练想要达到50%以上的正确率，最少需要的样本数量已达20w张，对电脑的显卡性能要求过高。

因此，**考虑从神经网络结构入手，重新构造更为复杂卷积神经网络，以减少样本量的需求，并尽量避免过拟合的问题**。同时，考虑到显卡本身的性能，准备**定义一个数据生成器在训练过程中利用CPU无限生成大量数据**。

## VER.2  Pred

### 数据集收集

因为本次尝试会构建更复杂的神经网络，考虑到电脑显卡的性能问题，如果将训练图片一次性生成，对电脑的负载过大。所以**定义了一个数据生成器**，在训练的过程中同时利用CPU生成大量数据**（随用随生，随生随删）所以不会真正占用存储空间，**故没有现成的数据集**。

这里直接利用了python已有的生成验证码的库captcha，每张图片由任意4个数字和大写英文字母组成，并同时加入随机不同颜色的若干噪点。代码如下：
```py
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


# In[3]:测试生成器


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
plt.imshow(X[0])
plt.title(decode(y))
```

**图片示例如下**：

![验证码图片样例](https://github.com/Zebra-zzzz/Verification-Code-Recognition/blob/master/VER.2%20Pred/output.png)

### 方法挑选

本实验采用Python软件作为英文文字识别的工具。

本次实验依然选用Keras构建CNN，鉴于VER.1 Verification最后出现了过拟合的情况，故参考了Keras官方文档的VGG16结构。

本次构建的深度卷积网络结构为：特征提取部分使用的是两个卷积，一个池化的结构。之后再将它Flatten，然后添加Dropout。因为每个图片对应的是四位字母和数字的组合，每个字符又对应36种可能性，所以最后连接四个分类器，每个分类器是36个神经元，分别输出36个字符的概率。

特征提取部分的结构参考了VGG16，来源：（[Keras官方文档](https://keras.io/applications/#vgg16)）



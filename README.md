# Verification-Code-Recognition
## About this project
这是《用户体验设计》这门课的第一个大作业，要求实现对验证码（英文字母+数字）的识别。作业示例使用了PIL库读取图片，并利用了光学字符识别包pytesseract，依赖tesseract ORC engine进行图像识别。但最后的识别率并不是很高，仅为22.8%，所以选择了目前更为主流的深度学习进行验证码的训练和验证。

完整的项目分为两个部分，分别对应一个验证版本。对于第一个版本，尽管用于训练的数据集最终的训练结果非常可观，但是在对剩下30张用于test的图片进行验证时，正确率却出奇的低，怀疑是出现了过拟合的问题。第二个版本对神经网络结构做了一定更改，重新构造了更为复杂卷积神经网络，以减少样本量的需求，并尽量避免过拟合的问题。同时，考虑到显卡本身的性能，准备定义一个数据生成器在训练过程中利用CPU无限生成大量数据。

___

## VER.1  Verification

### 数据集收集

因为本次识别打算利用深度学习框架Keras，用Python建立一个卷积神经网络进行训练，所以需要大量的数据集。
因此选择了直接通过以下代码(`/VER.1 Verification/genPic.py`)**自动生成24000张训练图片和30张验证图片**,全部放在`/VER.1 Verification/images`的文件夹内:
 ```
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
```
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

虽然Python现在已有较为常用的光学字符识别包pytesseract，但是最后的识别率并不是很高。所以选择了目前更为主流的深度学习进行验证码的训练和验证。查阅了相关资料后，CNN特别适用于对于图片的识别，故决定选用名为Keras的深度学习框架搭建CNN。在参考学习了Coursera上由deeplearning.ai开设的深度学习课程（[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)），具体为第四门课（[Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)）第二周目名为The Happy House的model（用于识别人脸是否开心），自行对应教学文档的格式写出了适用于验证码识别的程序，并修改了部分卷积神经网络的结构。


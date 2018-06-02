#!/usr/bin/python
#-*-coding:utf-8-*-
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

    

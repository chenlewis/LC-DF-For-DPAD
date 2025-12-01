from torch.utils import data
import os
from torchvision.transforms import transforms as T
from PIL import Image
import random
# import cv2
# from torch.utils.data import DataLoader
# from sklearn.model_selection import KFold
# import numpy as np
# import torch

class Copy_Detection(data.Dataset):
    def __init__(self, root,  transforms=None, train=True, test=False):
        super(Copy_Detection, self).__init__()
        self.test = test

        #加载路径
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs_0 = [os.path.join(root, '0', img) for img in os.listdir(os.path.join(root, '0'))]
        imgs_1 = [os.path.join(root, '1', img) for img in os.listdir(os.path.join(root, '1'))]
        # random.shuffle(imgs)
        # if self.test:
        #     imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1]))
        # # else:
        #     if len(imgs.split('-')) == 1:
        #         imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1]), reverse=True) #reverse=true 从高到低进行排序
        #     elif len(imgs.split('-')) == 2:
        #         imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1].split('-')[-2]), reverse=True)

        # print(imgs_num)
        # 划分训练集和测试集 验证：训练 = 1：9
        # if self.test:
        #     self.imgs = imgs[int(0.9*imgs_num):]
        # elif train:
        #     self.imgs = imgs[:int(0.8*imgs_num)]
        # else:
        #     self.imgs = imgs[int(0.8*imgs_num):int(0.9*imgs_num)]
        '''
        传统验证方法
        '''
        if self.test:
            # self.imgs = imgs
            self.imgs = imgs_0 + imgs_1
        elif train:
            self.imgs = imgs[:int(0.8*imgs_num)]
            # self.imgs = imgs
        else:
            self.imgs = imgs[int(0.8*imgs_num):]

        # imgs_num = len(imgs)
        '''
        交叉验证
        '''
        # if self.test:
        #     self.imgs = imgs
        # elif train:
        #     self.imgs = list(set(imgs) - set(imgs[int(0.2*i*imgs_num):int(0.2*(i+1)*imgs_num)]))
        # else:
        #     self.imgs = imgs[int(0.2*i*imgs_num):int(0.2*(i+1)*imgs_num)]

        #交叉验证
        # else:
        #     self.imgs = imgs
        # self.imgs = imgs #跨库训练

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
            )
            if self.test or not train:
                #验证和测试
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalize
                ])
            else:
                # 训练集
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    # T.CenterCrop(224),
                    T.RandomHorizontalFlip(p=0.5),  # 水平翻转
                    # T.RandomVerticalFlip(p=0.5),  # 垂直翻转
                    # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),  # 调整亮度、对比度、饱和度
                    # T.RandomErasing(p= 0.5, scale=(0.02,0.33), ratio=(0.3,0.3), value=0, inplace=True),
                    T.ToTensor(),
                    # 对图像进行随机遮挡, scale遮挡区域的面积，ratio遮挡区域长宽比,value设置遮挡区域的像素值
                    # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value='1234'),
                    normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # print(img_path)
        # if self.test:
        #     # label = 1 if len(img_path.split('/')[-1].split('_')) == 3 else 0
        #     label = img_path.split('.')[-2].split('/')[-1]  #测试集数据上的名称
        #     # print(label)
        # else:
        #     label = 1 if len(img_path.split('/')[-1].split('_')) == 7 else 0     #gai 3
        if self.test:
            label = int(os.path.basename(os.path.dirname(img_path)))  # Extract label from parent folder name
        else:
            label = int(os.path.basename(os.path.dirname(img_path)))  # Extract label from filename

        data = Image.open(img_path).convert("RGB")  # 强制转三通道
        # img = Image.open(path).convert("RGB")  # 强制转三通道
        #转换成灰度图像
        #data = Image.open(img_path).convert('')
        data = self.transforms(data)

        return data, label,img_path

    def __len__(self):
        return len(self.imgs)



class Copy_Detection_1(data.Dataset):
    # def __init__(self, img,  transforms=None, train=True, test=False, i=0):
    #     super(Copy_Detection_1, self).__init__()
    #     self.test = test
    #
    #     #加载路径
    #     # imgs = [os.path.join(root, img) for img in os.listdir(root)]
    #     imgs = img
    #     imgs_num = len(imgs)
    #
    #     '''
    #     传统验证方法
    #     '''
    #     if self.test:
    #         self.imgs = imgs
    #     elif train:
    #         self.imgs = imgs[:int(0.8*imgs_num)]
    #     else:
    #         self.imgs = imgs[int(0.8*imgs_num):]
    #
    #
    #     if transforms is None:
    #         normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225]
    #         )
    #         if self.test or not train:
    #             #验证和测试
    #             self.transforms = T.Compose([
    #                 T.Resize((224, 224)),
    #                 T.ToTensor(),
    #                 normalize
    #             ])
    #         else:
    #             # 训练集
    #             normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #             self.transforms = T.Compose([
    #                 T.Resize((224, 224)),
    #                 # T.CenterCrop(224),
    #                 # T.RandomHorizontalFlip(p=0.5),  # 水平翻转
    #                 # T.RandomVerticalFlip(p=0.5),  # 垂直翻转
    #                 # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),  # 调整亮度、对比度、饱和度
    #                 # T.RandomErasing(p= 0.5, scale=(0.02,0.33), ratio=(0.3,0.3), value=0, inplace=True),
    #                 T.ToTensor(),
    #                 # 对图像进行随机遮挡, scale遮挡区域的面积，ratio遮挡区域长宽比,value设置遮挡区域的像素值
    #                 # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value='1234'),
    #                 normalize
    #             ])
    #
    # def __getitem__(self, index):
    #     img_path = self.imgs[index]
    #     if self.test:
    #         # label = 1 if len(img_path.split('/')[-1].split('_')) == 3 else 0
    #         label = img_path.split('.')[-2].split('/')[-1]  #测试集数据上的名称
    #     else:
    #         label = 1 if len(img_path.split('/')[-1].split('_')) == 7 else 0    #gai
    #     data = Image.open(img_path)
    #     # print(data.size)
    #     # data = cv2.imread(img_path)
    #     # print(data.shape)
    #     #转换成灰度图像
    #     #data = Image.open(img_path).convert('')
    #     data = self.transforms(data)
    #     return data, label
    def __init__(self, root, transforms=None, train=True):
        super(Copy_Detection_1, self).__init__()
        self.train = train
        self.root = root

        # if self.train:
        #     self.imgs_0 = [os.path.join(root, '0/', img) for img in os.listdir(os.path.join(root, '0/'))]
        #     self.imgs_1 = [os.path.join(root, '1/', img) for img in os.listdir(os.path.join(root, '1/'))]
        #     self.imgs = self.imgs_0 + self.imgs_1
        # else:
        #     # self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        #     self.imgs_0 = [os.path.join(root, '0/', img) for img in os.listdir(os.path.join(root, '0/'))]
        #     self.imgs_1 = [os.path.join(root, '1/', img) for img in os.listdir(os.path.join(root, '1/'))]
        #     self.imgs = self.imgs_0 + self.imgs_1
        # imgs_0 = [os.path.join(root, '0', img) for img in os.listdir(os.path.join(root, '0')) if img.endswith('.jpg')]
        # imgs_1 = [os.path.join(root, '1', img) for img in os.listdir(os.path.join(root, '1')) if img.endswith('.jpg')]
        imgs_0 = [os.path.join(root, '0', img) for img in os.listdir(os.path.join(root, '0'))]
        imgs_1 = [os.path.join(root, '1', img) for img in os.listdir(os.path.join(root, '1'))]
        # if self.test:
        #     self.imgs = imgs_0 + imgs_1
        if self.train:
            self.imgs = imgs_0[:int(0.8 * len(imgs_0))] + imgs_1[:int(0.8 * len(imgs_1))]
        else:
            self.imgs = imgs_0[int(0.8 * len(imgs_0)):] + imgs_1[int(0.8 * len(imgs_1)):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if not self.train:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalize
                ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # print('imgs',img_path)

        if self.train:
            label = int(os.path.basename(os.path.dirname(img_path)))  # Extract label from parent folder name
            # print('label1',label)
        else:
            # label = int(os.path.basename(img_path).split('.')[0])  # Extract label from filename
            label = int(os.path.basename(os.path.dirname(img_path)))
            # print('label2',label)

        data = Image.open(img_path).convert("RGB")  # 强制转三通道
        data = self.transforms(data)
        return data, label



    def __len__(self):
        return len(self.imgs)


































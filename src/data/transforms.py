#!/usr/bin/env python3

"""Image transformations."""
import torchvision as tv
import torch
from PIL import Image, ImageFilter
import random
import numpy as np
import io

class AddIlluminationNoise(object):
    def __init__(self, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5), hue_range=(0.0, 0.05), apply_illumination_prob=0.5):
        """
        Args:
            brightness_range: 随机调整亮度的范围 [min_brightness, max_brightness]
            contrast_range: 随机调整对比度的范围 [min_contrast, max_contrast]
            saturation_range: 随机调整饱和度的范围 [min_saturation, max_saturation]
            hue_range: 随机调整色相的范围 [min_hue, max_hue] (注意必须是非负数)
            apply_illumination_prob: 应用光照噪声的概率
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range  # 确保 hue 是非负范围
        self.apply_illumination_prob = apply_illumination_prob

    def __call__(self, img):
        if random.random() < self.apply_illumination_prob:
            # 随机生成每次应用时的参数
            brightness = random.uniform(*self.brightness_range)
            contrast = random.uniform(*self.contrast_range)
            saturation = random.uniform(*self.saturation_range)
            hue = random.uniform(*self.hue_range)  # 确保 hue 是非负的

            # 创建 ColorJitter 实例并应用
            color_jitter = tv.transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            )
            return color_jitter(img)
        return img
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, apply_noise_prob=0.5):
        self.mean = mean
        self.std = std
        self.apply_noise_prob = apply_noise_prob

    def __call__(self, img):
        if random.random() < self.apply_noise_prob:
            np_img = np.array(img)
            noise = np.random.normal(self.mean, self.std, np_img.shape)
            noisy_img = np_img + noise
            noisy_img = np.clip(noisy_img, 0, 255)  # Clip to valid range
            return Image.fromarray(np.uint8(noisy_img))
        return img

class AddGaussianNoise0(object):
    def __init__(self, mean=0.0, std_range=(10, 30), apply_noise_prob=0.5):
        self.mean = mean
        self.std_range = std_range  # 允许随机选择 std，增加多样性
        self.apply_noise_prob = apply_noise_prob

    def __call__(self, img):
        if random.random() < self.apply_noise_prob:
            np_img = np.array(img, dtype=np.float32)

            # 生成不同通道的高斯噪声
            std_r = random.uniform(*self.std_range)  # R通道
            std_g = random.uniform(*self.std_range)  # G通道
            std_b = random.uniform(*self.std_range)  # B通道

            noise = np.stack([
                np.random.normal(self.mean, std_r, np_img[:, :, 0].shape),
                np.random.normal(self.mean, std_g, np_img[:, :, 1].shape),
                np.random.normal(self.mean, std_b, np_img[:, :, 2].shape)
            ], axis=-1)

            noisy_img = np_img + noise
            noisy_img = np.clip(noisy_img, 0, 255)  # 保持像素值在有效范围
            return Image.fromarray(np.uint8(noisy_img))

        return img

class AddBlur(object):
    def __init__(self, radius=2, apply_blur_prob=0.5):
        self.radius = radius
        self.apply_blur_prob = apply_blur_prob

    def __call__(self, img):
        if random.random() < self.apply_blur_prob:
            return img.filter(ImageFilter.GaussianBlur(self.radius))
        return img

class AddResize(object):
    def __init__(self, scale=1.0, apply_resize_prob=0.5):
        self.scale = scale
        self.apply_resize_prob = apply_resize_prob

    def __call__(self, img):
        if random.random() < self.apply_resize_prob:
            width, height = img.size
            new_size = (int(width * self.scale), int(height * self.scale))
            return img.resize(new_size, Image.ANTIALIAS)
        return img
class AddCompress(object):
    def __init__(self, quality_range=(50, 90), apply_compress_prob=0.5):
        """
        初始化压缩参数
        Args:
            quality_range (tuple): JPEG 压缩质量的范围 (min_quality, max_quality)。
            apply_compress_prob (float): 应用压缩的概率。
        """
        self.quality_range = quality_range
        self.apply_compress_prob = apply_compress_prob
    def __call__(self, img):
        """
        应用 JPEG 压缩
        Args:
            img (PIL.Image.Image or np.ndarray): 输入图像，可以是 PIL 图像或 NumPy 数组。
        Returns:
            PIL.Image.Image: 经过压缩后的图像。
        """
        if random.random() < self.apply_compress_prob:
            quality = random.randint(*self.quality_range)  # 随机选择质量
            buffer = io.BytesIO()
            # 如果是 NumPy 数组，转换为 PIL 图像
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            # 确保输入是 PIL 图像
            if not isinstance(img, Image.Image):
                raise ValueError("Input must be a PIL image or a NumPy array.")

            # 保存为压缩 JPEG 格式到内存缓冲区
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            # 从缓冲区加载图像并返回
            compressed_img = Image.open(buffer)
            return compressed_img.convert("RGB")  # 确保返回 RGB 格式
        return img
def get_transforms(split, size):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384

    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                AddIlluminationNoise(
                    brightness_range=(0.2, 0.4),
                    contrast_range=(0.2, 0.4),
                    saturation_range=(0.2, 0.4),
                    hue_range=(0.2, 0.4),  # hue 范围调整为非负
                    apply_illumination_prob=0.3
                ), # 0.2 0.4 0.3
                AddCompress(quality_range=(30, 60), apply_compress_prob=0.2),# 30 90 0.2
                AddGaussianNoise(mean=0, std=30.0, apply_noise_prob=0.2),  # 添加高斯噪声 0 30 0.2
                AddBlur(radius=2, apply_blur_prob=0.2),  # 添加模糊
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform

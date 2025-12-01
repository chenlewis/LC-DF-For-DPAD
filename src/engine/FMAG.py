import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import torch.fft

def RGB_fft_tensor(img_tensor):
    """
    Applies FFT on each channel of an RGB tensor.

    Args:
        img_tensor (torch.Tensor): Tensor of shape (C, H, W) representing the image in RGB.
                                   Expected to be in float format.

    Returns:
        fshift_origin_abs (torch.Tensor): Amplitude of the FFT result for each channel.
        fshift_angle (torch.Tensor): Phase of the FFT result for each channel.
    """
    # Ensure the input is a float tensor for FFT compatibility
    assert img_tensor.dtype in [torch.float32, torch.float64], "Input tensor must be a float tensor"
    # FFT on each channel
    fshift = torch.fft.fftshift(torch.fft.fft2(img_tensor, dim=(-2, -1)), dim=(-2, -1))
    # Separate amplitude and phase
    fshift_origin_abs = torch.abs(fshift)
    fshift_angle = torch.angle(fshift)
    return fshift_origin_abs, fshift_angle

def RGB_ifft_tensor(fshift_tensor):
    """
    Applies IFFT on each channel of an RGB tensor.

    Args:
        fshift_tensor (torch.Tensor): Tensor of shape (C, H, W) representing the FFT-transformed image.

    Returns:
        img_reconstructed (torch.Tensor): Reconstructed image tensor of shape (C, H, W).
    """
    # Perform inverse FFT shift
    ifft_input = torch.fft.ifftshift(fshift_tensor, dim=(-2, -1))
    # Perform IFFT and take the real part
    img_reconstructed = torch.abs(torch.fft.ifft2(ifft_input, dim=(-2, -1)))
    return img_reconstructed


def show(img):
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    fshift_abs = np.log(np.abs(fshift))
    plt.imshow(fshift_abs, 'gray')
    plt.show()


def add_noise_tensor(spectral, mask, alpha):
    """
    Add noise to the spectral tensor.

    Args:
        spectral (torch.Tensor): Tensor of shape (C, H, W) representing the spectral data.
        mask (torch.Tensor): Binary mask tensor of shape (C, H, W).
        a1, a2 (float): Range for random alpha values.
        mu, sigma (float): Mean and standard deviation for Gaussian noise.

    Returns:
        torch.Tensor: Spectral tensor with noise applied.
    """
    mask1 = 1 - mask
    spectral_mask_noise = (alpha * spectral) * mask + spectral * mask1
    # spectral_mask_noise = spectral * mask1
    return spectral_mask_noise


import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter

def find_peaks_from_magnitude(magnitude_spectrum, r=15):
    """
    自动检测幅度谱中的显著峰值点
    Args:
        magnitude_spectrum (np.array): 幅度谱（推荐使用对数变换后的结果）
        r (int): 边界排除半径
    Returns:
        positions: 检测到的峰值坐标列表[(x,y)]
    """
    # 最大值滤波增强局部极大值
    local_maxima = maximum_filter(magnitude_spectrum, size=30) ##30 0.6 30 0.2

    # 自适应阈值（取最大值的40%）
    threshold = 0.6*local_maxima  # 与第二段代码保持一致
    # 获取候选峰值点

    peaks = np.where((magnitude_spectrum == local_maxima) & (magnitude_spectrum > threshold))
    peaks = list(zip(peaks[1], peaks[0]))  # 转换为(x,y)坐标

    positions = []
    height, width = magnitude_spectrum.shape
    center = height // 2

    # 过滤无效点
    for (x, y) in peaks:
        # 排除中心点和边界附近点
        if (x == center and y == center) or \
                (x < r or x > width - r) or \
                (y < r or y > height - r):
            continue
        positions.append((x, y))

    return positions

def get_mask_tensor(img_fft, height=224, width=224):
    """
    自动生成频域掩码（PyTorch Tensor版本）
    Args:
        img_fft: 频域数据 (C,H,W) torch.Tensor
    Returns:
        mask: 频域掩码 (1,H,W) torch.Tensor
    """
    device = img_fft.device
    center = height // 2

    # 转换到CPU并取第一个通道
    fft_magnitude = torch.abs(img_fft[0]).cpu().numpy()
    log_spectrum = np.log(fft_magnitude + 1e-6)  # 对数变换

    # 自动检测峰值点
    positions = find_peaks_from_magnitude(log_spectrum)

    # 创建空掩码
    mask = np.zeros((height, width), dtype=np.uint8)

    # 在四个对称方向绘制掩码
    for (x, y) in positions:
        dx = x - center
        dy = y - center

        # 计算径向距离
        radius = int(np.hypot(dx, dy))

        # 在四个对称方向绘制圆形
        if radius > 5:  # 过滤过小的半径
            cv2.circle(mask, (center + radius, center), 10, 1, -1)
            cv2.circle(mask, (center - radius, center), 10, 1, -1)
            cv2.circle(mask, (center, center + radius), 10, 1, -1)
            cv2.circle(mask, (center, center - radius), 10, 1, -1)

    return torch.tensor(mask, device=device).float().unsqueeze(0)


import torch
import torchvision.transforms as tv

# 归一化和反归一化的常量
normalize = tv.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

def denormalize(img_tensor):
    """
    反归一化操作，恢复到原始像素值范围
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)
    return img_tensor * std[:, None, None] + mean[:, None, None]

def normalize_tensor(img_tensor):
    """
    归一化操作，转换为ViT输入所需的格式
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)
    return (img_tensor - mean[:, None, None]) / std[:, None, None]

def Moire_fag_tensor(img_tensor, a):
    """
    改进的摩尔纹生成函数（自动检测峰值版）
    Args:
        img_tensor: 输入图像 (C,H,W) torch.Tensor, 归一化后的张量
        a: 噪声系数
    """

    # 傅里叶变换
    f_abs, f_angle = RGB_fft_tensor(img_tensor)

    # 自动生成掩码
    mask = get_mask_tensor(f_abs)
    inv_mask = 1 - mask

    # 频域处理
    main_freq = f_abs * mask
    noise_freq = add_noise_tensor(f_abs * inv_mask, inv_mask, a)

    # 重构频域
    combined = (main_freq + noise_freq) * torch.exp(1j * f_angle)

    # 逆变换
    img_reconstructed = RGB_ifft_tensor(combined)

    return img_reconstructed

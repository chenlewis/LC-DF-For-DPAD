import torch.nn as nn

from timm.models.layers import to_2tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         # allow different input size
#         # B, C, H, W = x.shape
#         # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
#         # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
#         #print("start",x.size())    #start torch.Size([1, 3, 256, 256])
#         x = self.proj(x)           #flatten before torch.Size([1, 768, 16, 16])
#         if self.flatten:
#             #print("flatten before",x.size())       #flatten before torch.Size([1, 768, 16, 16])
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#             #print("flatten transpose",x.size())   #flatten transpose torch.Size([1, 256, 768])
#         x = self.norm(x)
#         #print("after",x.size())                #after torch.Size([1, 256, 768])
#         return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        # 确保输出为768通道的投影层
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # 输出: (B, 768, 16, 16)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 经过第一层卷积
        # print("x",x.shape)
        # 使用投影卷积来生成 patches
        x = self.proj(x)  # 输出形状: [B, embed_dim, grid_size[0], grid_size[1]]
        # print("x", x.shape)
        # x = nn.functional.interpolate(x, size=(16, 16), mode='nearest')  # 使输出为 (B, 768, 16, 16)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x

class PatchEmbed_prompt(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        # 确保输出为768通道的投影层
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # 输出: (B, 768, 16, 16)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 经过第一层卷积
        # print("x",x.shape)
        # 使用投影卷积来生成 patches
        x = self.proj(x)  # 输出形状: [B, embed_dim, grid_size[0], grid_size[1]]
        # print("x", x.shape)
        # x = nn.functional.interpolate(x, size=(16, 16), mode='nearest')  # 使输出为 (B, 768, 16, 16)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x


class PatchEmbed_Swin(nn.Module):
    def __init__(self, img_size=224,patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)  # 将patch_size转换为tuple，保证宽高一致
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 使用卷积层将图像划分为patch并映射到embed_dim空间
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 计算grid_size和num_patches
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # norm_layer 是一个可选的规范化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()  # 获取图像的高度和宽度



        # 补齐图像尺寸，使得它能被patch_size整除
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        # 卷积操作，将图像划分为patch并映射到embed_dim空间
        x = self.proj(x)  # B C Wh Ww，C是嵌入后的维度
        # print("no Normmmmmmmm")
        if self.norm is not None:
            # print("Normmmmmmmm")
            Wh, Ww = x.size(2), x.size(3)  # 获取每个patch的高度和宽度
            x = x.flatten(2).transpose(1, 2)  # 将patch展平为[B, num_patches, embed_dim]
            x = self.norm(x)  # 对所有的patch进行规范化
            # x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)  # 恢复到原图的大小

        return x


class PatchEmbed_filter0(nn.Module):
    """2D Image to Patch Embedding with Fixed Filter Input"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Projection layer to create patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Normalization layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, chosen_filter):
        """
        Args:
            x: Input tensor
            chosen_filter: The filter type selected externally (e.g., "gaussian", "log", or "gabor")
        """
        # Apply the chosen filter
        if chosen_filter == "gaussian":
            x = self.apply_gaussian_filter(x)
        elif chosen_filter == "log":
            x = self.apply_log_filter(x)
        elif chosen_filter == "gabor":
            x = self.apply_gabor_filter(x)

        # Generate patches using the projection layer
        x = self.proj(x)  # Output shape: [B, embed_dim, grid_size[0], grid_size[1]]

        # Flatten if required
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x

    def apply_gaussian_filter(self, x, sigma=1.0):
        kernel = self.create_gaussian_kernel(x.shape[-1], sigma).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # Expand channel dimension
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def apply_log_filter(self, x, sigma=1.0):
        kernel = self.create_log_kernel(x.shape[-1], sigma).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # Expand channel dimension
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def apply_gabor_filter(self, x, wavelength=4, orientation=0.0):
        kernel = self.create_gabor_kernel(x.shape[-1], wavelength, orientation).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # Expand channel dimension
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """ Create Gaussian kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)

    @staticmethod
    def create_log_kernel(kernel_size, sigma):
        """ Create Laplacian of Gaussian (LoG) kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        log = (xx**2 + yy**2 - 2 * sigma**2) * gaussian / (sigma**4)
        log = log - log.mean()
        return log.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_gabor_kernel(kernel_size, wavelength, orientation):
        """ Create Gabor kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        orientation = torch.tensor(orientation, dtype=torch.float32)
        x_theta = xx * torch.cos(orientation) + yy * torch.sin(orientation)
        y_theta = -xx * torch.sin(orientation) + yy * torch.cos(orientation)
        gaussian = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * wavelength ** 2))
        sinusoid = torch.cos(2 * torch.pi * x_theta / wavelength)
        gabor = gaussian * sinusoid
        return gabor.unsqueeze(0).unsqueeze(0)

class FilterSelector0(nn.Module):
    """Shared Learnable Filter Selector"""
    def __init__(self):
        super().__init__()
        # Learnable probabilities for selecting a filter
        self.p_gaussian = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.p_log = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.p_gabor = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self):
        # Normalize probabilities to sum to 1
        filter_probs = torch.softmax(torch.stack([self.p_gaussian, self.p_log, self.p_gabor]), dim=0)
        filters = ["gaussian", "log", "gabor"]

        # Randomly select a filter based on the learnable probabilities
        chosen_filter = random.choices(filters, weights=filter_probs.tolist(), k=1)[0]
        return chosen_filter


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import pywt
import random
import numpy as np
import torch

from .torch_wavelets import DWT_2D,IDWT_2D

class FilterSelector(nn.Module):
    """Shared Learnable Filter Selector"""
    def __init__(self):
        super().__init__()
        # Learnable probabilities for selecting a filter
        self.p_gaussian = nn.Parameter(torch.tensor(0.4), requires_grad=True)
        # self.p_evp = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.p_wavelet = nn.Parameter(torch.tensor(0.6), requires_grad=True)

    def forward(self):
        # Normalize probabilities to sum to 1
        filter_probs = torch.softmax(torch.stack([self.p_gaussian, self.p_wavelet]), dim=0)
        filters = ["evp", "wavelet"]


        # Randomly select a filter based on the learnable probabilities
        chosen_filter = random.choices(filters, weights=filter_probs.tolist(), k=1)[0]
        return chosen_filter
#

class PatchEmbed_filter(nn.Module):
    """2D Image to Patch Embedding with Integrated Wavelet Dropout"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, wavelet='haar', norm_layer=None, flatten=True, wavelet_dropout_p=0.2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.wavelet_dropout_p = wavelet_dropout_p

        # Projection layer to create patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Normalization layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Wavelet transform module
        self.dwt = DWT_2D(wavelet)  # Discrete Wavelet Transform
        self.idwt = IDWT_2D(wavelet)  # Inverse Discrete Wavelet Transform

    def forward(self, x, chosen_filter=None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            chosen_filter: The filter type selected externally (e.g., "gaussian", "evp", or "wavelet").
                           If None, no filter will be applied.
        """
        # Generate patches using the projection layer
        x = self.proj(x)  # Output shape: [B, embed_dim, grid_size[0], grid_size[1]]
        if chosen_filter!=None:
        # Apply wavelet dropout if specified
            if chosen_filter == "evp":
                x = self.apply_wavelet_dropout(x)
            if chosen_filter == "wavelet":
                x = self.apply_wavelet_dropout(x)

        # Flatten if required
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x

    def apply_wavelet_dropout(self, x):
        """
        Apply wavelet dropout using the provided DWT module.
        """
        # Perform discrete wavelet transform (returns concatenated tensor)
        x_dwt = self.dwt(x)
        # print("x",x.shape)
        # print("x_dwt",x_dwt.shape)
        # Manually split concatenated tensor into LL, LH, HL, HH
        B, C, H, W = x_dwt.shape
        H, W = H // 2, W // 2
        LL = x_dwt[:, :, :H, :W]
        LH = x_dwt[:, :, :H, W:]
        HL = x_dwt[:, :, H:, :W]
        HH = x_dwt[:, :, H:, W:]
        # LL, LH, HL, HH = self.dwt(x)
        # print("LL",LL.shape)
        # print("LH",LH.shape)
        # print("HL",HL.shape)
        # print("HH",HH.shape)
        # Generate dropout masks for each high-frequency component
        dropout_mask_LH = (torch.rand(B, C, 1, 1, device=LH.device) > self.wavelet_dropout_p).float() / (
                    1 - self.wavelet_dropout_p)
        dropout_mask_HL = (torch.rand(B, C, 1, 1, device=HL.device) > self.wavelet_dropout_p).float() / (
                    1 - self.wavelet_dropout_p)
        dropout_mask_HH = (torch.rand(B, C, 1, 1, device=HH.device) > self.wavelet_dropout_p).float() / (
                    1 - self.wavelet_dropout_p)

        # Apply dropout to high-frequency components
        LH = LH * dropout_mask_LH
        HL = HL * dropout_mask_HL
        HH = HH * dropout_mask_HH
        # Now concatenate LL, LH, HL, HH back together to form the complete x_dwt
        x_dwt = torch.cat([LL, LH, HL, HH], dim=1)  # Concatenate along the channel dimension

        # Reconstruct the tensor back to the original size using inverse DWT
        x_filtered = self.idwt(x_dwt)  # Now using the complete x_dwt to reconstruct
        # Reconstruct the tensor back to the original size using inverse DWT
        # x_filtered = self.idwt((LL, LH, HL, HH))

        return x_filtered
    def apply_gaussian_filter(self, x, sigma=1.0, kernel_size=5):
            """
            Apply Gaussian filter using PyTorch.
            """
            def gaussian_kernel(size, sigma):
                coords = torch.arange(size) - size // 2
                grid = coords.repeat(size).view(size, size)
                kernel = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
                kernel /= kernel.sum()
                return kernel.view(1, 1, size, size)

            device = x.device
            kernel = gaussian_kernel(kernel_size, sigma).to(device)
            padding = kernel_size // 2
            x_filtered = F.conv2d(x, kernel.expand(x.size(1), 1, -1, -1), padding=padding, groups=x.size(1))
            return x_filtered

# class PatchEmbed_filter(nn.Module):
#     """2D Image to Patch Embedding with Integrated Filters"""
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, wavelet='haar', norm_layer=None, flatten=True):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size // patch_size, img_size // patch_size)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten
#
#         # Projection layer to create patches
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#         # Normalization layer
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#         # Wavelet transform module
#         self.dwt = DWT_2D(wavelet)  # Discrete Wavelet Transform
#         self.idwt = IDWT_2D(wavelet)  # Inverse Discrete Wavelet Transform
#
#     def forward(self, x, chosen_filter=None):
#         """
#         Args:
#             x: Input tensor (B, C, H, W)
#             chosen_filter: The filter type selected externally (e.g., "gaussian", "evp", or "wavelet").
#                            If None, no filter will be applied.
#         """
#         # Generate patches using the projection layer
#         x = self.proj(x)  # Output shape: [B, embed_dim, grid_size[0], grid_size[1]]
#
#         # Apply frequency dropout (filtering) if a filter is specified
#         if chosen_filter is not None:
#             if chosen_filter == "gaussian":
#                 x = self.apply_gaussian_filter(x, sigma=2.0, kernel_size=5)
#             elif chosen_filter == "evp":
#                 x = self.apply_evp_filter(x, rate=16)
#             elif chosen_filter == "wavelet":
#                 x = self.apply_wavelet_filter(x)
#
#         # Flatten if required
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#
#         x = self.norm(x)
#         return x
#
#     def apply_gaussian_filter(self, x, sigma=1.0, kernel_size=5):
#         """
#         Apply Gaussian filter using PyTorch.
#         """
#         def gaussian_kernel(size, sigma):
#             coords = torch.arange(size) - size // 2
#             grid = coords.repeat(size).view(size, size)
#             kernel = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
#             kernel /= kernel.sum()
#             return kernel.view(1, 1, size, size)
#
#         device = x.device
#         kernel = gaussian_kernel(kernel_size, sigma).to(device)
#         padding = kernel_size // 2
#         x_filtered = F.conv2d(x, kernel.expand(x.size(1), 1, -1, -1), padding=padding, groups=x.size(1))
#         return x_filtered
#
#     def apply_evp_filter(self, x, rate=16):
#         """
#         Apply EVP high-frequency enhancement filter to the input tensor using PyTorch.
#         """
#         device = x.device
#         b, c, h, w = x.shape
#         mask = torch.zeros((b, c, h, w), device=device)
#         line = max(1, int((h * w * rate) ** 0.5 // 2))
#         mask[:, :, h // 2 - line:h // 2 + line, w // 2 - line:w // 2 + line] = 1
#
#         # Apply FFT
#         fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"), dim=(-2, -1))
#         fft = fft * (1 - mask)  # Apply high-pass filter (1-mask)
#
#         # Inverse FFT
#         filtered = torch.fft.ifftshift(fft, dim=(-2, -1))
#         filtered = torch.fft.ifft2(filtered, norm="forward").real
#         return torch.abs(filtered)
#
#     def apply_wavelet_filter(self, x):
#         """
#         Apply wavelet transform for high-frequency enhancement using the provided DWT module.
#         """
#         # Perform discrete wavelet transform
#         LL, LH, HL, HH = self.dwt(x)
#         # Keep only the high-frequency components (LH, HL, HH)
#         high_freq = torch.cat([LH, HL, HH], dim=1)  # Concatenate high-frequency components
#         # Reconstruct the tensor back to the original size using inverse DWT
#         x_filtered = self.idwt(high_freq)
#
#         # Ensure the output matches the target shape (B, C, 768)
#         B, C, H, W = x_filtered.shape
#         if H * W != 768:
#             # Adjust the spatial dimensions to match the required 768
#             x_filtered = x_filtered.flatten(2).transpose(1, 2)  # BCHW -> BNC
#             x_filtered = x_filtered[:, :, :768]  # Trim or pad to match 768
#             x_filtered = x_filtered.transpose(1, 2).view(B, C, H, W)  # BNC -> BCHW (restore)
#
#         return x_filtered


class PatchEmbedWithDropout(PatchEmbed):
    def __init__(self, *args, low_dropout_prob=0.5, high_dropout_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_low_dropout = HighLowFrequencyDropout(low_dropout_prob, high_dropout_prob)

    def forward(self, x):
        x = self.proj(x)  # 投影到嵌入维度
        x = self.high_low_dropout(x)  # 应用高低频 Dropout
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # 展平为序列形式
        x = self.norm(x)
        return x

class HighLowFrequencyDropout(nn.Module):
    """ High/Low Frequency Dropout """
    def __init__(self, low_dropout_prob=0.5, high_dropout_prob=0.5):
        super(HighLowFrequencyDropout, self).__init__()
        self.low_dropout_prob = low_dropout_prob
        self.high_dropout_prob = high_dropout_prob
        self.dynamic_filter = DynamicFilterChannel(inchannels=768)  # Replace inchannels with proper value

    def forward(self, x):
        # Split into high and low frequencies
        low_part, high_part = self.dynamic_filter(x)

        # Drop low frequency
        if self.training and torch.rand(1).item() < self.low_dropout_prob:
            low_part = torch.zeros_like(low_part)

        # Drop high frequency
        if self.training and torch.rand(1).item() < self.high_dropout_prob:
            high_part = torch.zeros_like(high_part)

        # Combine frequencies
        return low_part + high_part

class PatchEmbedWithFrequencyDropout(nn.Module):
    """ 2D Image to Patch Embedding with Shared Frequency and Filter Dropout """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = self.to_2tuple(img_size)
        patch_size = self.to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Conv2D for patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Normalization layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Unified Frequency and Filter Dropout
        self.freq_filter_dropout = UnifiedFrequencyFilterDropout(embed_dim=embed_dim)

    def to_2tuple(self, x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, x, shared_choice):
        # Patch embedding with Conv2D
        x = self.proj(x)
        # Apply combined frequency and filter dropout with shared choice
        x = self.freq_filter_dropout(x, shared_choice)
        # Flatten to (B, num_patches, embed_dim) if needed
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # Normalize embeddings
        x = self.norm(x)
        return x


class UnifiedFrequencyFilterDropout(nn.Module):
    """ Unified High/Low Frequency and Filter Dropout with External Shared Selection """
    def __init__(self, embed_dim=768):
        super(UnifiedFrequencyFilterDropout, self).__init__()
        # Dynamic filter module
        self.dynamic_filter = DynamicFilterChannel(inchannels=embed_dim)

    def forward(self, x, shared_choice):
        # Split into high and low frequencies
        low_part, high_part = self.dynamic_filter(x)

        # Apply operation based on shared choice
        if shared_choice == "high":
            return high_part
        elif shared_choice == "low":
            return low_part
        elif shared_choice == "gaussian":
            return self.apply_gaussian_filter(x)
        elif shared_choice == "log":
            return self.apply_log_filter(x)
        elif shared_choice == "gabor":
            return self.apply_gabor_filter(x)
        else:
            return x

    def apply_gaussian_filter(self, x, sigma=1.0):
        kernel = self.create_gaussian_kernel(x.shape[-1], sigma).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # Expand channels
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def apply_log_filter(self, x, sigma=1.0):
        kernel = self.create_log_kernel(x.shape[-1], sigma).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # Expand channels
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def apply_gabor_filter(self, x, wavelength=4, orientation=0.0):
        kernel = self.create_gabor_kernel(x.shape[-1], wavelength, orientation).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # Expand channels
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """ Create Gaussian kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_log_kernel(kernel_size, sigma):
        """ Create Laplacian of Gaussian (LoG) kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        log = (xx**2 + yy**2 - 2 * sigma**2) * gaussian / (sigma**4)
        log = log - log.mean()
        return log.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_gabor_kernel(kernel_size, wavelength, orientation):
        """ Create Gabor kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        orientation = torch.tensor(orientation, dtype=torch.float32)
        x_theta = xx * torch.cos(orientation) + yy * torch.sin(orientation)
        y_theta = -xx * torch.sin(orientation) + yy * torch.cos(orientation)
        gaussian = torch.exp(-(x_theta**2 + y_theta**2) / (2 * wavelength**2))
        sinusoid = torch.cos(2 * torch.pi * x_theta / wavelength)
        gabor = gaussian * sinusoid
        return gabor.unsqueeze(0).unsqueeze(0)


# class FilterDropout0(nn.Module):
#     """ Apply random filters (Gaussian, LoG, Gabor) as Dropout """
#     def __init__(self, p_gaussian=0.3, p_log=0.3, p_gabor=0.4):
#         super(FilterDropout, self).__init__()
#         self.p_gaussian = p_gaussian
#         self.p_log = p_log
#         self.p_gabor = p_gabor
#
#     def forward(self, x):
#         if not self.training:
#             return x
#
#         # 随机选择滤波器
#         filters = ["gaussian", "log", "gabor"]
#         chosen_filter = random.choices(filters, weights=[self.p_gaussian, self.p_log, self.p_gabor], k=1)[0]
#
#         if chosen_filter == "gaussian":
#             return self.apply_gaussian_filter(x)
#         elif chosen_filter == "log":
#             return self.apply_log_filter(x)
#         elif chosen_filter == "gabor":
#             return self.apply_gabor_filter(x)
#         return x
#
#     def apply_gaussian_filter(self, x, sigma=1.0):
#         kernel = self.create_gaussian_kernel(x.shape[-1], sigma).to(x.device)
#         kernel = kernel.expand(x.size(1), 1, -1, -1)  # 扩展通道
#         return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))
#
#     def apply_log_filter(self, x, sigma=1.0):
#         kernel = self.create_log_kernel(x.shape[-1], sigma).to(x.device)
#         kernel = kernel.expand(x.size(1), 1, -1, -1)  # 扩展通道
#         return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))
#
#     def apply_gabor_filter(self, x, wavelength=4, orientation=0.0):
#         kernel = self.create_gabor_kernel(x.shape[-1], wavelength, orientation).to(x.device)
#         kernel = kernel.expand(x.size(1), 1, -1, -1)  # 扩展通道
#         return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))
#
#     @staticmethod
#     def create_gaussian_kernel(kernel_size, sigma):
#         """ Create Gaussian kernel """
#         k = kernel_size // 2
#         x = torch.arange(-k, k + 1, dtype=torch.float32)
#         y = torch.arange(-k, k + 1, dtype=torch.float32)
#         xx, yy = torch.meshgrid(x, y)
#         kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
#         kernel = kernel / kernel.sum()
#         return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
#
#     @staticmethod
#     def create_log_kernel(kernel_size, sigma):
#         """ Create Laplacian of Gaussian (LoG) kernel """
#         k = kernel_size // 2
#         x = torch.arange(-k, k + 1, dtype=torch.float32)
#         y = torch.arange(-k, k + 1, dtype=torch.float32)
#         xx, yy = torch.meshgrid(x, y)
#         gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
#         log = (xx**2 + yy**2 - 2 * sigma**2) * gaussian / (sigma**4)
#         log = log - log.mean()
#         return log.unsqueeze(0).unsqueeze(0)
#
#     @staticmethod
#     def create_gabor_kernel(kernel_size, wavelength, orientation):
#         """ Create Gabor kernel """
#         k = kernel_size // 2
#         x = torch.arange(-k, k + 1, dtype=torch.float32)
#         y = torch.arange(-k, k + 1, dtype=torch.float32)
#         xx, yy = torch.meshgrid(x, y)
#         # 确保 orientation 是 Tensor 类型
#         orientation = torch.tensor(orientation, dtype=torch.float32)
#         # 使用 Tensor 计算
#         x_theta = xx * torch.cos(orientation) + yy * torch.sin(orientation)
#         y_theta = -xx * torch.sin(orientation) + yy * torch.cos(orientation)
#         gaussian = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * wavelength ** 2))
#         sinusoid = torch.cos(2 * torch.pi * x_theta / wavelength)
#         gabor = gaussian * sinusoid
#         return gabor.unsqueeze(0).unsqueeze(0)


class DynamicFilterChannel0(nn.Module):
    """ Dynamic filter channel module for splitting high and low frequencies """

    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(DynamicFilterChannel, self).__init__()
        self.kernel_size = kernel_size
        self.group = group
        self.stride = stride

        # Convolution layers for generating dynamic filters
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group * kernel_size ** 2, group * kernel_size ** 2, kernel_size=1, stride=1,
                                   bias=False)
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        # Padding for convolution
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        # Adaptive average pooling
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x  # Save the original input for high-frequency computation

        # Generate low-frequency filter
        low_filter = self.ap(x)  # Adaptive average pooling
        low_filter = self.conv(low_filter)  # Convolution to generate dynamic filters
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))  # Apply gating
        low_filter = self.bn(low_filter)  # Batch normalization

        # Unfold the input tensor to prepare for dynamic filtering
        n, c, h, w = x.shape
        unfolded_x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            n, self.group, c // self.group, self.kernel_size ** 2, h * w
        )
        # Reshape and normalize the low-frequency filter
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)  # Apply softmax across the kernel dimensions
        # Apply dynamic filtering to compute the low-frequency part
        low_part = torch.sum(unfolded_x * low_filter, dim=3).reshape(n, c, h, w)
        # Compute the high-frequency part
        high_part = identity - low_part
        return low_part, high_part

class DynamicFilterChannel(nn.Module):
    """ Dynamic filter channel module for splitting high and low frequencies """
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(DynamicFilterChannel, self).__init__()
        self.kernel_size = kernel_size
        self.group = group
        self.conv = nn.Conv2d(inchannels, group * kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group * kernel_size**2)

    def forward(self, x):
        identity = x
        low_filter = self.conv(F.adaptive_avg_pool2d(x, (1, 1)))
        low_filter = self.act_gate(low_filter)
        low_filter = self.bn(low_filter)
        # High and low frequency split
        n, c, h, w = x.shape
        low_part = torch.zeros_like(x)  # Dummy implementation for simplicity
        high_part = identity - low_part
        return low_part, high_part

class PatchEmbedWithFrequencyDropout2(nn.Module):
    """ 2D Image to Patch Embedding with High/Low Frequency and Filter Dropout """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 low_dropout_prob=0.5, high_dropout_prob=0.5, p_gaussian=0.3, p_log=0.3, p_gabor=0.4):
        super().__init__()
        img_size = self.to_2tuple(img_size)
        patch_size = self.to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        # Conv2D for patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Normalization layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # High/Low Frequency Dropout
        self.high_low_dropout = HighLowFrequencyDropout(low_dropout_prob, high_dropout_prob)
        # Filter Dropout
        self.filter_dropout = FilterDropout(p_gaussian, p_log, p_gabor)

    def to_2tuple(self, x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, x):
        # Patch embedding with Conv2D
        x = self.proj(x)
        # Apply High/Low Frequency Dropout
        x = self.high_low_dropout(x)
        # Apply Filter Dropout
        x = self.filter_dropout(x)
        # Flatten to (B, num_patches, embed_dim) if needed
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # Normalize embeddings
        x = self.norm(x)
        return x

class HighLowFrequencyDropout(nn.Module):
    """ High/Low Frequency Dropout """
    def __init__(self, low_dropout_prob=0.5, high_dropout_prob=0.5):
        super(HighLowFrequencyDropout, self).__init__()
        self.low_dropout_prob = low_dropout_prob
        self.high_dropout_prob = high_dropout_prob
        self.dynamic_filter = DynamicFilterChannel(inchannels=768)  # Replace inchannels with proper value

    def forward(self, x):
        # Split into high and low frequencies
        low_part, high_part = self.dynamic_filter(x)

        # Drop low frequency
        if self.training and torch.rand(1).item() < self.low_dropout_prob:
            low_part = torch.zeros_like(low_part)

        # Drop high frequency
        if self.training and torch.rand(1).item() < self.high_dropout_prob:
            high_part = torch.zeros_like(high_part)

        # Combine frequencies
        return low_part + high_part

class FilterDropout(nn.Module):
    """ Apply random filters (Gaussian, LoG, Gabor) as Dropout """
    def __init__(self, p_gaussian=0.3, p_log=0.3, p_gabor=0.4):
        super(FilterDropout, self).__init__()
        self.p_gaussian = p_gaussian
        self.p_log = p_log
        self.p_gabor = p_gabor

    def forward(self, x):
        if not self.training:
            return x

        # 随机选择滤波器
        filters = ["gaussian", "log", "gabor"]
        chosen_filter = random.choices(filters, weights=[self.p_gaussian, self.p_log, self.p_gabor], k=1)[0]

        if chosen_filter == "gaussian":
            return self.apply_gaussian_filter(x)
        elif chosen_filter == "log":
            return self.apply_log_filter(x)
        elif chosen_filter == "gabor":
            return self.apply_gabor_filter(x)
        return x

    def apply_gaussian_filter(self, x, sigma=1.0):
        kernel = self.create_gaussian_kernel(x.shape[-1], sigma).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # 扩展通道
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def apply_log_filter(self, x, sigma=1.0):
        kernel = self.create_log_kernel(x.shape[-1], sigma).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # 扩展通道
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def apply_gabor_filter(self, x, wavelength=4, orientation=0.0):
        kernel = self.create_gabor_kernel(x.shape[-1], wavelength, orientation).to(x.device)
        kernel = kernel.expand(x.size(1), 1, -1, -1)  # 扩展通道
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """ Create Gaussian kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)

    @staticmethod
    def create_log_kernel(kernel_size, sigma):
        """ Create Laplacian of Gaussian (LoG) kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        log = (xx**2 + yy**2 - 2 * sigma**2) * gaussian / (sigma**4)
        log = log - log.mean()
        return log.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_gabor_kernel(kernel_size, wavelength, orientation):
        """ Create Gabor kernel """
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        # 确保 orientation 是 Tensor 类型
        orientation = torch.tensor(orientation, dtype=torch.float32)
        # 使用 Tensor 计算
        x_theta = xx * torch.cos(orientation) + yy * torch.sin(orientation)
        y_theta = -xx * torch.sin(orientation) + yy * torch.cos(orientation)
        gaussian = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * wavelength ** 2))
        sinusoid = torch.cos(2 * torch.pi * x_theta / wavelength)
        gabor = gaussian * sinusoid
        return gabor.unsqueeze(0).unsqueeze(0)

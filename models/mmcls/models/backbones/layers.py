import torch.nn as nn

from timm.models.layers import to_2tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 计算网格大小
        self.num_patches = self.grid_size ** 2  # 计算总 patch 数量

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        输入: x [B, 3, H, W]
        输出: x [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # 卷积投影
        x = x.flatten(2).transpose(1, 2)  # [B, embed_dim, num_patches] → [B, num_patches, embed_dim]
        return x

class Mlp(nn.Module):
    """ Multilayer Perceptron (MLP) used in Transformer Blocks """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # 默认 4 倍扩展

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        输入: x [B, num_patches, embed_dim]
        输出: x [B, num_patches, embed_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """ DropPath (Stochastic Depth) """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        输入: x [B, num_patches, embed_dim]
        输出: x [B, num_patches, embed_dim] （随机丢弃部分路径）
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适配所有维度
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

import math
import torch.nn.functional as F

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """ Truncated normal initialization (用于 ViT 权重初始化) """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    l, u = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor

def lecun_normal_(tensor):
    """ LeCun 正态初始化（用于 MLP 层）"""
    fan_in = tensor.shape[1]  # fan_in = 输入特征数
    std = 1.0 / math.sqrt(fan_in)
    return trunc_normal_(tensor, mean=0., std=std)

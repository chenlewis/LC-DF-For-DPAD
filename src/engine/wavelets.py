import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super(DWT_2D, self).__init__()
        # 获取小波滤波器（Haar 或其他类型）
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  # 高通滤波器
        dec_lo = torch.Tensor(w.dec_lo[::-1])  # 低通滤波器

        # 构建滤波器的二维版本
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  # LL (低频)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  # LH (水平高频)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  # HL (垂直高频)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # HH (对角高频)

        # 添加维度以匹配卷积输入
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))  # LL 滤波器
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))  # LH 滤波器
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))  # HL 滤波器
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))  # HH 滤波器

    def forward(self, x):
        """
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            LL, LH, HL, HH: 小波分解后的四个子带
        """
        # 通道分组卷积，每个通道都应用小波滤波器
        x_ll = F.conv2d(x, self.w_ll.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        x_lh = F.conv2d(x, self.w_lh.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        x_hl = F.conv2d(x, self.w_hl.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        x_hh = F.conv2d(x, self.w_hh.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))

        return x_ll, x_lh, x_hl, x_hh


class IDWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)  # 高通重构滤波器
        rec_lo = torch.Tensor(w.rec_lo)  # 低通重构滤波器

        # 构建重构滤波器的二维版本
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)  # LL (低频重构)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)  # LH (水平高频重构)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)  # HL (垂直高频重构)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)  # HH (对角高频重构)

        # 添加维度以匹配卷积输入
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))  # LL 滤波器
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))  # LH 滤波器
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))  # HL 滤波器
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))  # HH 滤波器

    def forward(self, LL, LH, HL, HH):
        """
        Args:
            LL, LH, HL, HH: 小波变换的 4 个子带，形状为 (B, C, H//2, W//2)
        Returns:
            x_reconstructed: 逆小波变换后的重建图像，形状为 (B, C, H, W)
        """
        # 反卷积以进行逆小波变换
        B, C, H, W = LL.shape
        H, W = H * 2, W * 2  # 恢复到原始尺寸

        # 使用反卷积将子带合成为原始图像
        x_ll = F.conv_transpose2d(LL, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        x_lh = F.conv_transpose2d(LH, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hl = F.conv_transpose2d(HL, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hh = F.conv_transpose2d(HH, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)

        # 将四个重构分量相加得到重建的图像
        x_reconstructed = x_ll + x_lh + x_hl + x_hh

        return x_reconstructed

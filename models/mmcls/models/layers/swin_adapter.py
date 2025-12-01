import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

import math
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

# from .attn import Attention
from .adapter import Bi_direct_adapter,Single_adapter,MonaAdapter,Single_adapter_s

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .deit import selector,Attention_cfged,Mlp_cfged
import numpy as np

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Adjust relative position bias table size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).chunk(3, dim=0)

        q = q.squeeze(0) * self.scale
        attn = (q @ k.squeeze(0).transpose(-2, -1))

        # Adjust relative position bias handling to match new L
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v.squeeze(0)).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlockAdapter(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_dim=16, scale_factor=8):
        """
        Swin Transformer Adapter Block with window attention and adapter layers.
        """
        super(SwinTransformerBlockAdapter, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # Normalization layers
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # print("dim",dim)
        # Attention layer with window mechanism
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # Adapter layers
        self.adapter_st = Single_adapter_s(dim,scale_factor)
        self.adapter_t = Bi_direct_adapter(dim,adapter_dim)
        self.adapter_t2 = Bi_direct_adapter(dim, adapter_dim)
        # self.adapter_st_up = nn.Linear(dim // scale_factor, dim)
        # self.adapter_t_up = nn.Linear(dim // scale_factor, dim)

    def forward(self, x_FMAG, x, z, mask_x=None, mask_z=None):
        """
        Forward pass for Swin Transformer Adapter block with window attention computation and adapter integration.
        """
        xor = x
        zor = z
        mask_x = None
        # mask_z = None
        B, L, C = x.shape
        # print("x.shape",x.shape)
        # Dynamically calculate H and W from the shape of the input x
        H = int(L ** 0.5)
        W = H  # Assuming the input is a square (L = H * W)
        assert L == H * W, "Input feature has wrong size"

        # Adapter integration (after attention and before FFN)

        if x_FMAG is not None:
            # print("x_FMAG.shape",x_FMAG.shape)
            x_FMAG =self.adapter_st(x_FMAG)
            # x_FMAG = self.norm1(x_FMAG)
            x = x_FMAG + x
        shortcut_x = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift for x
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        #     # attn_mask_x = mask_x
        #     attn_mask_x = None
        # else:
        #     shifted_x = x
        #     attn_mask_x = None
        shifted_x = x
        attn_mask_x = None
        # Partition windows for x
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA for x
        attn_windows_x = self.attn(x_windows, mask=attn_mask_x)  # nW*B, window_size*window_size, C

        # Merge windows for x
        attn_windows_x = attn_windows_x.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_x, self.window_size, Hp, Wp)  # B H' W' C

        # Reverse cyclic shift for x
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Now handle z modality
        shortcut_z = z
        z = self.norm1(z)
        z = z.view(B, H, W, C)

        # Pad feature maps for z to multiples of window size
        z = F.pad(z, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp_z, Wp_z, _ = z.shape

        # Cyclic shift for z
        if self.shift_size > 0:
            shifted_z = torch.roll(z, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            if mask_z is not None:
                Hp = int(np.ceil(H / self.window_size)) * self.window_size  # 使用 self.window_size 替代 window_size
                Wp = int(np.ceil(W / self.window_size)) * self.window_size  # 使用 self.window_size 替代 window_size
                img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 创建一个全零的掩码
                # 定义水平和垂直切片，以便扫描整个图像
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt  # 每个窗口区域分配一个唯一的标识符
                        cnt += 1
                # 划分窗口
                mask_windows = window_partition(img_mask, self.window_size)  # 划分为多个窗口
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # 将窗口展平
                # 计算注意力掩码
                attn_mask_z = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 计算每对窗口之间的关系
                attn_mask_z = attn_mask_z.masked_fill(attn_mask_z != 0, float(-100.0)).masked_fill(attn_mask_z == 0,
                                                                                          float(0.0))  # 填充掩码
                # 打印掩码形状
                # print("attn_mask:", attn_mask_z.shape)
            # attn_mask_z = mask_z
            # print("attn_mask_z",attn_mask_z.shape)
        else:
            shifted_z = z
            attn_mask_z = None

        # Partition windows for z
        z_windows = window_partition(shifted_z, self.window_size)  # nW*B, window_size, window_size, C
        z_windows = z_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA for z
        attn_windows_z = self.attn(z_windows, mask=attn_mask_z)  # nW*B, window_size*window_size, C

        # Merge windows for z
        attn_windows_z = attn_windows_z.view(-1, self.window_size, self.window_size, C)
        shifted_z = window_reverse(attn_windows_z, self.window_size, Hp_z, Wp_z)  # B H' W' C

        # Reverse cyclic shift for z
        if self.shift_size > 0:
            z = torch.roll(shifted_z, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            z = shifted_z

        if pad_r > 0 or pad_b > 0:
            z = z[:, :H, :W, :].contiguous()

        z = z.view(B, H * W, C)

        # Adapter integration (after attention and before FFN)
        x = shortcut_x + self.drop_path(x)+ self.drop_path(self.adapter_t(shortcut_z))
        # x = shortcut_x + self.drop_path(x)
        z = shortcut_z + self.drop_path(z)+ self.drop_path(self.adapter_t(shortcut_x))
        # z = shortcut_z + self.drop_path(z)
        identity_x = x
        x = self.norm2(x)
        # FFN (Feed-forward Network) for x
        x = self.mlp(x)
        identity_z = z
        z = self.norm2(z)
        # FFN (Feed-forward Network) for z
        z = self.mlp(z)
        # Final adapter integration for x
        x = identity_x + self.drop_path(x) + self.drop_path(self.adapter_t2(identity_z))
        # x = identity_x + self.drop_path(x)
        # Final adapter integration for z
        z = identity_z + self.drop_path(z) + self.drop_path(self.adapter_t2(identity_x))
        # z = identity_z + self.drop_path(z)
        return x, attn_windows_x, z, attn_windows_z


class SwinTransformerAdapter(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7,shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., adapter_dim=16, scale_factor=8):
        super(SwinTransformerAdapter, self).__init__()

        # Create the stages with SwinTransformerAdapter blocks
        self.stage1 = self._make_stage(embed_dim, num_heads[0], depths[0], window_size,shift_size, mlp_ratio, drop, attn_drop,
                                       drop_path, adapter_dim, scale_factor)
        self.stage2 = self._make_stage(embed_dim * 2, num_heads[1], depths[1], window_size,shift_size, mlp_ratio, drop, attn_drop,
                                       drop_path, adapter_dim, scale_factor)
        self.stage3 = self._make_stage(embed_dim * 4, num_heads[2], depths[2], window_size,shift_size, mlp_ratio, drop, attn_drop,
                                       drop_path, adapter_dim, scale_factor)
        self.stage4 = self._make_stage(embed_dim * 8, num_heads[3], depths[3], window_size,shift_size, mlp_ratio, drop, attn_drop,
                                       drop_path, adapter_dim, scale_factor)

        # Patch Merging layer for downsampling after each stage
        self.patch_merging1 = PatchMerging(embed_dim, norm_layer=nn.LayerNorm)
        self.patch_merging2 = PatchMerging(embed_dim * 2, norm_layer=nn.LayerNorm)
        self.patch_merging3 = PatchMerging(embed_dim * 4, norm_layer=nn.LayerNorm)

    def _make_stage(self, dim, num_heads, depth, window_size,shift_size, mlp_ratio, drop, attn_drop, drop_path, adapter_dim,
                    scale_factor):
        layers = []
        for _ in range(depth):
            layers.append(SwinTransformerBlockAdapter(
                dim=dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,shift_size=shift_size,
                qkv_bias=True, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                adapter_dim=adapter_dim, scale_factor=scale_factor
            ))
        return nn.Sequential(*layers)

    def forward(self, x_FMAG, x, z, mask_x=None, mask_z=None):
        # Pass through the stages
        for block in self.stage1:
            x, _, z, _ = block(x_FMAG, x, z, mask_x, mask_z)

        # Patch Merge at the end of stage1 to prepare for stage2
        x = self.patch_merging1(x)
        if x_FMAG is not None:
            x_FMAG = self.patch_merging1(x_FMAG)
        z = self.patch_merging1(z)

        for block in self.stage2:
            x, _, z, _ = block(x_FMAG, x, z, mask_x, mask_z)

        # Patch Merge at the end of stage2 to prepare for stage3
        x = self.patch_merging2(x)
        if x_FMAG is not None:
         x_FMAG = self.patch_merging2(x_FMAG)
        z = self.patch_merging2(z)

        for block in self.stage3:
            x, _, z, _ = block(x_FMAG, x, z, mask_x, mask_z)

        # Patch Merge at the end of stage3 to prepare for stage4
        x = self.patch_merging3(x)
        if x_FMAG is not None:
            x_FMAG = self.patch_merging3(x_FMAG)
        z = self.patch_merging3(z)

        for block in self.stage4:
            x, _, z, _ = block(x_FMAG, x, z, mask_x, mask_z)

        # Return the final output
        return x, z

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H = int(L**0.5)
        W = H
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# Helper functions for window partition and reverse
def window_partition(x, window_size):
    """
    Partition input into windows of size (window_size, window_size).
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse the window partition to reconstruct the feature map.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H: Height of original feature map
        W: Width of original feature map
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
#
# class SwinTransformerBlock(nn.Module):
#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#
#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
#         )
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             act_layer(),
#             nn.Dropout(drop),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )
#
#     def forward(self, x, mask_matrix):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "Input feature has wrong size"
#
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#
#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#
#         # partition windows
#         x_windows = shifted_x.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
#         x_windows = x_windows.contiguous().view(-1, self.window_size * self.window_size, C)
#
#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=mask_matrix)
#
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = attn_windows.permute(0, 3, 1, 2).contiguous()
#
#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#
#         x = x.view(B, H * W, C)
#
#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
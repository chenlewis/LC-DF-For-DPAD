import math
import torch
import torch.nn as nn
# from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from .attn import Attention
from .adapter import Bi_direct_adapter, Single_adapter, MonaAdapter

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .deit import selector, Attention_cfged, Mlp_cfged


def tokens_to_image(x, height, width):
    """
    将 ViT 的 token 表示重塑为 2D 图像表示。
    Args:
        x: 输入特征 [B, N, D]
        height: 图像的高度
        width: 图像的宽度
    Returns:
        reshaped_x: 转换后的 4D 特征 [B, D, H, W]
    """
    batch_size, num_tokens, embed_dim = x.size()
    cls_token = x[:, 0, :]  # 提取 CLS token
    patch_tokens = x[:, 1:, :]  # 提取 patch token
    patch_tokens = patch_tokens.transpose(1, 2).contiguous()  # [B, D, N]
    reshaped_x = patch_tokens.view(batch_size, embed_dim, height, width)  # [B, D, H, W]
    return reshaped_x, cls_token


def image_to_tokens(x, cls_token):
    """
    将 2D 图像表示还原为 ViT 的 token 表示。
    Args:
        x: 输入特征 [B, D, H, W]
        cls_token: CLS token [B, 1, D]
    Returns:
        tokens: 转换后的 token 表示 [B, N, D]
    """
    batch_size, embed_dim, height, width = x.size()
    patch_tokens = x.view(batch_size, embed_dim, -1)  # [B, D, N]
    patch_tokens = patch_tokens.transpose(1, 2).contiguous()  # [B, N, D]
    tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)  # [B, N, D]
    return tokens

class CEABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, adapter_dim=16,
                 scale_factor=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 使用 DropPath 或 nn.Identity

        self.keep_ratio_search = keep_ratio_search
        self.adap_st = Single_adapter(scale_factor)
        self.adap_st_p = Single_adapter(scale_factor)
        self.adap_t = Bi_direct_adapter(adapter_dim)
        # self.adap_t_p = Bi_direct_adapter(adapter_dim)
        self.adap2_t = Bi_direct_adapter(adapter_dim)

    def forward(self, x_otsu, z_otsu, x, z, x_origin, z_origin, adapter_up, i, global_index_template,
                global_index_templatei, global_index_search, global_index_searchi,
                mask_x=None, mask_z=None, ce_template_mask=None, keep_ratio_search=None):
        xori = x
        xor = x
        zor = z
        mask_x = None
        mask_z = None
        # print("x", x.shape)
        # print("xi",xi.shape)
        x_attn, attn = self.attn(self.norm1(x), mask_x, True)
        # print(self.adap_t(self.norm1(xori)).shape)
        # print(self.adap_t(self.norm1(xi)).shape)

        x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(zor))  #########-------------------------adapter
        # x = x + self.drop_path(x_attn)
        # lens_z_new = global_index_template.shape[1]
        z_attn, i_attn = self.attn(self.norm1(z), mask_z, True)
        # z = z + self.drop_path(z_attn)
        z = z + self.drop_path(z_attn) + self.drop_path(self.adap_t(xor))  #########-------------------------adapter
        # print("z.shape", z.shape)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))  ###-------adapter
        xori = x
        # x = x + self.drop_path(self.mlp(self.norm2(x)))    ###-------adapter
        # x = x + self.drop_path(self.mlp(self.norm2(x)))+ self.drop_path(self.adap2_t(self.norm2(z)))+self.drop_path(
        #     adapter_up(self.adap_st(self.norm1(z_origin)))) ###-------adapter
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))  ###-------adapter
        # z = z + self.drop_path(self.mlp(self.norm2(z)))+self.drop_path(self.adap2_t(self.norm2(xori))) ###-------adapter
        z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(
            self.adap2_t(self.norm2(xori))) + self.drop_path(
            adapter_up(self.adap_st_p(self.norm1(z_origin))))
        return x, attn, z, i_attn


import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CEABlock_FMAG(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, adapter_dim=16,
                 scale_factor=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # 融合和差异建模模块
        # self.diff_mlp = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        # )
        # 使用 DropPath 或 nn.Identity
        # self.keep_ratio_search = keep_ratio_search
        self.adap_st = Single_adapter(dim, scale_factor)
        # self.adap_st_p = Single_adapter(scale_factor)
        self.adap_t = Bi_direct_adapter(dim, adapter_dim)
        # self.adap_t_p = Bi_direct_adapter(adapter_dim)
        self.adap2_t = Bi_direct_adapter(dim, adapter_dim)

    def forward(self, x_otsu, z_otsu, x_FMAG, x, z, x_origin, z_origin, adapter_up, adapter_up2, i,
                global_index_template,
                global_index_templatei, global_index_search, global_index_searchi, mask_x=None,
                mask_z=None, ce_template_mask=None, keep_ratio_search=None):
        xor = x
        zor = z
        mask_x = None
        # mask_z = None
        # z_origin = adapter_up2(self.adap_st_p(self.norm2(z_origin)))
        # z = torch.cat((
        #         z[:, :1, :],
        #         z_origin + z[:, 1:, :],
        #     ), dim=1)

        # 通过适配器进一步更新 x 和 z
        if x_FMAG is not None:
            x_FMAG = adapter_up(self.adap_st(self.norm2(x_FMAG)))
            x = torch.cat((
                x[:, :1, :],
                x_FMAG + x[:, 1:, :],
            ), dim=1)
            x_attn, attn = self.attn(self.norm1(x), mask_x, True)
            # 更新 x 通过 DropPath 和适配器
            x = x + self.drop_path(x_attn) + self.drop_path(
                self.adap_t(self.norm1(zor)))  #########-------------------------adapter
            # x = x + self.drop_path(x_attn)
            # lens_z_new = global_index_template.shape[1]
            z_attn, i_attn = self.attn(self.norm1(z), mask_z, True)
            # z = z + self.drop_path(z_attn)
            z = z + self.drop_path(z_attn) + self.drop_path(
                self.adap_t(self.norm1(xor)))  #########-------------------------adapter
            xori = x
            # cls_token = x[:, 0:1, :]  # 提取 cls_token，假设它是第一个位置的 token
            # cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)  # 定义全 0 的 cls_token，维度与原始一致
            # 将 x_FMAG 和 z_origin 合并到 x 中
            # x_FMAG = torch.cat([cls_token, x_FMAG], dim=1)  # 将 cls_token 与 x_FMAG 拼接
            # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))+self.drop_path(self.adap_st(self.norm2(x_FMAG)))
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))

        else:
            x_attn, attn = self.attn(self.norm1(x), mask_x, True)
            # 更新 x 通过 DropPath 和适配器
            # x = x + self.drop_path(x_attn)+ self.drop_path(self.adap_t(self.norm1(zor))) #########-------------------------adapter
            x = x + self.drop_path(x_attn)
            z_attn, i_attn = self.attn(self.norm1(z), mask_z, True)
            z = z + self.drop_path(z_attn)
            # z = z + self.drop_path(z_attn) + self.drop_path(self.adap_t(self.norm1(xor)))
            xori = x
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))

        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))
        z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(self.adap2_t(self.norm2(xori)))
        return x, attn, z, i_attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_all_tokens:
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])
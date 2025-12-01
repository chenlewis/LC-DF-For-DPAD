import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from ..layers.utils import combine_tokens, recover_tokens, token2feature, feature2token
from ..layers.vit import VisionTransformer
from ..layers.patch_embed import PatchEmbed, PatchEmbed_prompt, PatchEmbedWithFrequencyDropout, PatchEmbed_filter, PatchEmbed_Swin
from ..layers.attn_blocks import CEBlock, candidate_elimination_adapter

# from ..layers.attn_adapt_blocks import CEABlock   ##BAT
# from ..layers.attn_adapt_blocks import CEABlock_EVP as CEABlock  ##BAT
# from ..layers.swin_adapter import SwinTransformerAdapter as CEABlock  ##BAT
from ..layers.swin_adapter import SwinTransformerAdapter as STAblk
from ..layers.dualstream_attn_blocks import DSBlock  ## Dual Stream without adapter
from ..layers.adapter import Bi_direct_adapter

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
from ..builder import BACKBONES
import math
import logging
import warnings
import errno
import os
import sys
import re
import zipfile
from urllib.parse import urlparse  # noqa: F401

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
_logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
import torchvision as tv
from src.utils import logging

loggers = logging.get_logger("visual_prompt")
from collections import OrderedDict
import random
from difflib import get_close_matches
import numpy as np
import cv2
from skimage import filters
from PIL import Image
import timm

def load_dict_params(state_dict, model, matched_keys, weights):
    """
    递归加载字典类型的参数。
    """
    for k, v in state_dict.items():
        # 如果v是字典类型，递归处理
        if isinstance(v, dict):
            # print(f"发现字典类型参数: {k}")
            # 递归调用，处理字典内部的参数
            load_dict_params(v, model, matched_keys, weights)
        else:
            # 如果v是tensor类型，继续正常处理
            close_matches = get_close_matches(k, [i for i in model.state_dict()], n=1, cutoff=0.2)

            if close_matches:
                best_match = close_matches[0]  # 选择最接近的匹配
                model_param = model.state_dict()[best_match]  # 获取当前模型的参数

                # 检查形状是否匹配，并且当前键没有被匹配过
                if model_param.shape == v.shape and best_match not in matched_keys:
                    # print(f"匹配的层：{k} (模型层：{best_match})，形状：{v.shape}")
                    weights[best_match] = v  # 使用state_dict中的参数
                    matched_keys.add(best_match)  # 标记当前键为已匹配

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
    if 'pretrained_finetune' in cfg and cfg['pretrained_finetune']:
        try:
            state_dict = torch.load(cfg['pretrained_finetune'], map_location='cpu')  # 或者使用 'cuda:0'
        except Exception as e:
            _logger.error(f"加载预训练权重失败 {cfg['pretrained_finetune']}: {e}")
            return
    else:
        state_dict = load_state_dict_from_url(cfg['url'], progress=False, map_location='cpu')
        print('从imagenet21k加载预训练权重')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    weights = OrderedDict()  # 存放权重
    matched_keys = set()

    # 加载字典类型的参数
    load_dict_params(state_dict, model, matched_keys, weights)

    # 对于未匹配的层，使用模型的默认权重
    for i in model.state_dict():
        if i not in matched_keys:
            weights[i] = model.state_dict()[i]  # 使用模型中的默认参数
            # print(f"未匹配层：{i}")
    if in_chans == 1:
        print("here 111")
        conv1_name = cfg['first_conv']
        _logger.info(
            'Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        print("here 333")
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I == 3:
            _logger.warning(
                'Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            _logger.info(
                'Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[
                :, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for segformer trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    model.load_state_dict(weights, strict=True)
    # 加载 swin_base_patch4_window7_224 预训练模型
    # for name, param in model.named_parameters():
    #     if name in weights:
    #         print(f"层 {name} 成功加载")
    # else:
    #     print(f"层 {name} 未加载")


def _cfg(url='', pretrained_finetune=None, **kwargs):
    return {
        'url': url,
        'pretrained_finetune': pretrained_finetune,
        'num_classes': 2, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }


# /home/lyj/code_lpq/Output_RODHQ/BAT-CMA/5e5/32_16_16/3ada/model_epoch_1.pth /home/lyj/code_lpq/pretrain_finetune/vit_base_p16_224-4e355ebd.pth
default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'swin_base_patch4_window7_224_pre': _cfg(url=None,  # 可以是URL，也可以是None,
                                 pretrained_finetune='/home/lyj/code_lpq/Output_RODHQ/BAT-FMAG/Swin/MF162/mask/3/0model_epoch_0.pth', ),
    'swin_base_patch4_window7_224_FMAG': _cfg(url=None,  # 可以是URL，也可以是None,
                                      pretrained_finetune='/home/lyj/MFM/SwinB_FAG.pth', ),
    'swin_base_patch4_window7_224': _cfg(url=None,  # 可以是URL，也可以是None,
                                         pretrained_finetune='/home/lyj/code_lpq/pretrain_finetune/swin_base_patch4_window7_224.pth', ),

}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@BACKBONES.register_module()
class SwinBatCMA(nn.Module):
    def __init__(self, model_name='', img_size=224, patch_size=16, patch_size_prompt=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None,
                 template_size=None, new_patch_size=None, adapter_type=None, pretrain=True, add_cls_token=True,
                 adapter_dim=16, scale_factor=8, window_size=7):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        # super().__init__()
        super(SwinBatCMA, self).__init__()
        # 保存 Swin Adapter 的参数

        self.window_size = window_size
        self.depth = depth  # 层数

        self.model_name = model_name
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed_Swin(
             patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,norm_layer=nn.LayerNorm,)
        # self.patch_embed_prompt_CMA = PatchEmbed_prompt(
        #     img_size=img_size, patch_size=patch_size_prompt, in_chans=in_chans, embed_dim=embed_dim)
        # self.patch_embed_prompt_rgb = PatchEmbed_prompt(
        #     img_size=img_size, patch_size=patch_size_prompt, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt_FMAG = PatchEmbed_Swin(
            patch_size=patch_size_prompt, in_chans=in_chans, embed_dim=embed_dim,norm_layer=nn.LayerNorm,)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.add_cls_token = add_cls_token
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size

        self.num_patches_search = new_P_H * new_P_W
        # self.num_patches_search = 400  #289
        H, W = template_size
        new_P_H, new_P_W = H // patch_size_prompt, W // patch_size_prompt
        self.num_patches_template = new_P_H * new_P_W
        print(f"num_patches_template: {self.num_patches_template}")
        print('search', self.num_patches_search)
        print(self.num_patches_template)
        # self.num_patches_template = 400
        """add here, no need use backbone.finetune_track """  #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        # for i in range(depth):
        #     ce_keep_ratio_i = 1.0
        #     if ce_loc is not None and i in ce_loc:
        #         ce_keep_ratio_i = ce_keep_ratio[ce_index]
        #         ce_index += 1
        #     if i < 20:
        #         # blocks.append(
        #         #     CEABlock(
        #         #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        #         #         attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
        #         #         adapter_dim=adapter_dim, scale_factor=scale_factor)
        #         # )
        #         blocks.append(CEABlock(
        #                 dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
        #                 drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[i],norm_layer=norm_layer,
        #                 act_layer=act_layer,adapter_dim=adapter_dim,scale_factor=scale_factor,
        #                 window_size=7,shift_size=3  # 如果需要，可以设置shift_size为0（或者其他值）
        #             )
        #         )
        #     else:
        #         blocks.append(
        #             DSBlock(
        #                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        #                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
        #                 keep_ratio_search=ce_keep_ratio_i)
        #         )

        # self.blocks = nn.Sequential(*blocks)
        self.blocks = STAblk(num_classes=2,img_size=224, patch_size=4,embed_dim=embed_dim,
                     depths=[2, 2, 18, 2],num_heads=[4, 8, 16, 32],window_size=7,shift_size=3,mlp_ratio=4.,qkv_bias=True,
                     drop=0.1,attn_drop=0.1,drop_path=0.1, adapter_dim=adapter_dim, scale_factor=scale_factor)

        self.norm = norm_layer(embed_dim)
        # self.init_weights(pretrain)
        pretrain = pretrain
        # self.load_pretrain(pretrain=pretrain)
        self.init_weights(pretrain=pretrain)
        # self.init_weights(pretrain=False)
        # self.adapter_up = nn.Linear(128 // scale_factor, 128)
        # self.adapter_up2 = nn.Linear(128 // scale_factor, 128)
        # self.adapter_up2 = None
        # self.adapter_up = None
        self.fusion_layer = None

    def init_weights(self, pretrain):
        print("loading")
        # nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.cls_token)
        # pretrain = True
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if pretrain:
            self.default_cfg = default_cfgs[self.model_name]
            loggers.info(f'Loaded pre-trained weights from {self.model_name}')
            # self.default_cfg = cfg
            loggers.info(f'Loaded pre-trained weights from {self.default_cfg["pretrained_finetune"]}')
            # if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
            #     loggers.info(f"------Loading pretrained model {self.model_name}")
            #     load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
            #                     num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, filter_fn=self._conv_filter)
            # else:
            #     loggers.info(f"----***Loading pretrained model {self.model_name}")
            #     load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
            #                     num_patches=self.patch_embed.num_patches, align_corners=self.align_corners)
            # loggers.info(f"----***Loading pretrained model {self.model_name}")
            load_pretrained(self, self.default_cfg, num_classes=self.num_classes, in_chans=self.in_chans)
        else:
            print('Initialize weight randomly')

    def forward_features(self, x_otsu, z_otsu, x_FMAG, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # 仅在训练模式下选择过滤器
        chosen_filter = None
        # if self.training:  # 判断是否在训练模式
        #     chosen_filter = self.filter_selector()
        #     print(chosen_filter)
        if x_FMAG is not None:
            # x_FMAG = self.patch_embed(x_FMAG, chosen_filter)
            x_FMAG = self.patch_embed_prompt_FMAG(x_FMAG)

        x, z = x_rgb, z_rgb
        # xi, zi = x_dte, z_dte
        # print("input x",x.size())
        # print("input z", z.size())
        zi = self.patch_embed(z)
        # zi = self.patch_embed(z,chosen_filter)
        # xi = self.patch_embed_prompt(x)
        # x = self.patch_embed(x)
        x = self.patch_embed(x)
        # print("input x",x.size())
        # print("input z", z.size())
        # x = self.patch_embed(x, chosen_filter)
        # z = self.patch_embed(z)
        z_otsu = zi
        x_otsu = x

        x_origin = None
        zi_origin = zi
        ###################################################################===========
        # attention mask handling
        # B, H, W
        # print("mask_x",mask_x)
        # print("mask_z", mask_z)
        if mask_z is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            # Create a cls_token mask of shape [batch_size, 1]
            cls_token_mask = torch.zeros(mask_z.shape[0], 1, dtype=torch.bool, device=mask_z.device)
            # cls_token_mask = torch.ones(mask_z.shape[0], 1, dtype=torch.bool, device=mask_z.device)
            # Ensure that both cls_token_mask and mask_z have the same number of dimensions
            if mask_z.dim() == 3:
                mask_z = mask_z.flatten(1)  # Flatten spatial dimensions [B, 256]
            # Concatenate the cls_token_mask to mask_z, ensuring consistent dimensions
            mask_z = torch.cat([cls_token_mask, mask_z], dim=1)  # [B, 257] after adding CLS token
        if mask_x is not None:
            # 进行插值操作，将 mask_z 的空间尺寸调整到适当大小
            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            # 展平到 [16, 256] 的形状
            mask_x = mask_x.flatten(1)  # 变为 [B, 256]
            # 填充 CLS token 的位置，调整为 [16, 257]
            cls_token_mask = torch.zeros(mask_x.shape[0], 1, dtype=torch.bool, device=mask_x.device)
            mask_x = torch.cat([cls_token_mask, mask_x], dim=1)  # 在第一个位置添加 CLS token 的 mask，变为 [16, 257]
        # if mask_z is not None:
        #     # 进行插值操作，将 mask_z 的空间尺寸调整到适当大小
        #     # mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
        #     # 展平到 [16, 256] 的形状
        #     # mask_z = mask_z.flatten(1)  # 变为 [B, 256]
        #     # 填充 CLS token 的位置，调整为 [16, 257]
        #     # Ensure mask_z has the same resolution as image after patch division
        #     print("mask_z.shape",mask_z.shape)
        #     mask_z = F.interpolate(mask_z[None].float(), size=(H // self.window_size, W // self.window_size),
        #                            mode='nearest')
        #     mask_z = mask_z[0].bool()  # Convert to boolean
        #     # print("mask_z", mask_z.shape)  # 应该是 [16, 257]
        # if mask_x is not None:
        #     # 进行插值操作，将 mask_z 的空间尺寸调整到适当大小
        #     mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
        #     # 展平到 [16, 256] 的形状
        #     mask_x = mask_x.flatten(1)  # 变为 [B, 256]
        #     # 填充 CLS token 的位置，调整为 [16, 257]
        #     cls_token_mask = torch.zeros(mask_x.shape[0], 1, dtype=torch.bool, device=mask_x.device)
        #     mask_x = torch.cat([cls_token_mask, mask_x], dim=1)  # 在第一个位置添加 CLS token 的 mask，变为 [16, 257]
        # mask_z = torch.cat([mask_x, mask_z], dim=1)
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # z += self.pos_embed_x
        # zi += self.pos_embed_z
        # x += self.pos_embed_x
        # xi += self.pos_embed_z
        # z_otsu += self.pos_embed_z
        # x_otsu += self.pos_embed_x

        # if self.add_cls_token:
        #     x = torch.cat([cls_tokens, x], dim=1)
        #     # z = torch.cat([cls_tokens, z], dim=1)
        #     zi = torch.cat([cls_tokens, zi], dim=1)
        #     # zi = torch.cat([cls_tokens, zi], dim=1)
        #     # z_otsu = torch.cat([cls_tokens, z_otsu], dim=1)
        #     # x_otsu = torch.cat([cls_tokens, x_otsu], dim=1)
        if x_FMAG is not None:
            # x_FMAG += self.pos_embed_x
            # x_FMAG = torch.cat([cls_tokens, x_FMAG], dim=1)
            x_FMAG = self.pos_drop(x_FMAG)
        x = self.pos_drop(x)
        # z = self.pos_drop(z)
        zi = self.pos_drop(zi)
        # x_otsu = self.pos_drop(x_otsu)
        # z_otsu = self.pos_drop(z_otsu)
        # zi = z_otsu
        removed_indexes_s = []

        # zi_origin = zi
        # removed_indexes_si = []
        removed_flag = False
        use_depth = False

        # 在调用 blk 时只传入必要的参数
        # for i, blk in enumerate(self.blocks):
        #     if use_depth:  # 仅在需要深度分量时使用
        #         x, global_index_t, global_index_s, removed_index_s, attn, \
        #             xi, global_index_ti, global_index_si, removed_index_si, attn_i = \
        #             blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x,
        #                 ce_template_mask, ce_keep_rate)
        #     else:
        #         # x, global_index_t, global_index_s, removed_index_s, attn, \
        #         #     z, global_index_ti, global_index_si, removed_index_si, attn_i = \
        #         #     blk(x, z, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x,
        #         #         ce_template_mask, ce_keep_rate)
        #         # OCR
        #         x, attn, zi, attn_i = blk(x_otsu, z_otsu, x_FMAG, x, zi, x_origin, zi_origin, adapter_up, mask_x,
        #                                   mask_z,ce_template_mask, ce_keep_rate)  ##FMAG
                # x, attn, zi, attn_i = blk(x_otsu,z_otsu,x,zi,x_origin,zi_origin,adapter_up,i, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x,mask_z,
                #         ce_template_mask, ce_keep_rate)
                # x, zi = blk(x,z_otsu,z_otsu,x_otsu)
        x, zi = self.blocks(x_FMAG, x, zi, mask_x,mask_z)
        x = torch.cat([x, zi], dim=1)
        # if x_FMAG is not None:
        #     x = torch.cat([x_FMAG,x], dim=1)
        # x = torch.cat([x, zi], dim=-1)  # 按特征维度拼接，x 和 z 的形状均为 [B, 257, D]
        # 拼接后 x 的形状为 [B, 257, 2*D]

        # re-concatenate with the template, which may be further used by other modules
        # if self.add_cls_token:
        #     x = torch.cat([cls_tokens, x], dim=1)
        # x = torch.cat([z, x], dim=1)
        # print("===========final out: ",x.size())
        # x = x + zi
        # x = torch.cat([x, xi], dim=2)
        # x = self.adap_headcat(x)

        # print("-------",x.shape)
        # print("attn",attn.size())
        # print(attn.shape)
        # 假设 x 和 zi 在 dim=1 上进行拼接，因此我们在相同的维度上拼接 attn 和 attn_i
        # attn_combined = torch.cat((attn.mean(dim=1), attn_i.mean(dim=1)), dim=1)  # 将多头注意力平均并拼接
        # print("attn", attn_combined.size())
        # print(attn.shape)
        # aux_dict = {
        #     "attn": attn_combined,
        #     "removed_indexes_s": removed_indexes_s,  # used for visualization
        # }
        aux_dict = None

        return x, aux_dict

    # def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
    #             tnc_keep_rate=None,
    #             return_last_attn=False):
    #
    #     x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, )
    #     print("x",x.shape)
    #     print("aux_dict", aux_dict.shape)
    #     return x, aux_dict
    def forward(self, x, x_FMAG=None, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
        z = self.IIC_sort(x)
        z = z.to(x.device)
        x_stsu, z_otsu, mask, mask_reverse = self.otsu_image(x, z)
        # 将所有变量移到与 x 相同的设备
        # x_stsu = x_stsu.to(x.device)
        # z_otsu = z_otsu.to(x.device)
        # mask = mask.to(x.device)
        # mask_reverse = mask_reverse.to(x.device)
        # mask = None
        # mask_reverse = None
        # ce_template_mask = mask_reverse
        x_stsu, z_otsu = x, z
        # x,aux_dict = self.forward_features(x_stsu,z_otsu,z, x,mask_z=None,mask_x=None, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, )
        # print("x",x.shape)
        x, aux_dict = self.forward_features(x_stsu, z_otsu, x_FMAG, z, x, mask_z=mask_reverse, mask_x=mask,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate, )
        # print("aux_dict", aux_dict)
        # return x, aux_dict
        return x

    def inverse_normalize(self, tensor, mean, std):
        """+
        对给定的张量进行逆归一化。

        Args:
            tensor (torch.Tensor): 归一化后的张量，通常是[B, C, H, W]。
            mean (list or tuple): 用于归一化的均值。
            std (list or tuple): 用于归一化的标准差。

        Returns:
            torch.Tensor: 逆归一化后的张量。
        """
        # 将 mean 和 std 转换为张量
        device = tensor.device  # 获取 tensor 的设备
        mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(std, device=device).view(1, -1, 1, 1)

        # 进行逆归一化
        tensor = tensor * std + mean
        return tensor

    def IIC_sort(self, x):
        # x should be in range 0-1
        # 逆归一化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x_inverse = self.inverse_normalize(x, mean, std)
        # print("x", x)
        # print("x_inverse", x_inverse)
        b_channel, g_channel, r_channel = x_inverse.unbind(dim=1)

        # 计算色度
        total = r_channel + g_channel + b_channel
        epsilon = 1e-7  # 一个非常小的值
        total += epsilon  # 避免除零

        c_b = b_channel / total
        c_g = g_channel / total
        c_r = r_channel / total

        # 标准化色度
        c_mean = torch.mean(torch.stack([c_b, c_g, c_r]), dim=0)
        c_std = torch.std(torch.stack([c_b, c_g, c_r]), dim=0)

        # 避免除以零
        c_std = torch.where(c_std == 0, torch.tensor(epsilon).to(c_std.device), c_std)

        c_b_norm = (c_b - c_mean) / c_std
        c_g_norm = (c_g - c_mean) / c_std
        c_r_norm = (c_r - c_mean) / c_std

        # 合并通道
        c_img = torch.stack((c_b_norm, c_g_norm, c_r_norm), dim=1)
        c_img_scaled = torch.zeros_like(c_img)

        for i in range(c_img.shape[0]):
            c_img_min = c_img[i].min()
            c_img_max = c_img[i].max()
            c_img_range = c_img_max - c_img_min
            c_img_range = torch.where(c_img_range == 0, torch.ones_like(c_img_range), c_img_range)
            c_img_scaled[i] = (c_img[i] - c_img_min) / c_img_range

        # 标准化
        normalize = tv.transforms.Normalize(
            mean=mean,
            std=std
        )

        c_img_normalized = normalize(c_img_scaled)
        c_img = c_img_normalized
        # print("c_img",c_img)
        # c_img = x
        # Normalize c_img to the range [-1, 1]
        # c_img_min = c_img.min(dim=1, keepdim=True)[0]
        # c_img_max = c_img.max(dim=1, keepdim=True)[0]
        # c_img = 2 * (c_img - c_img_min) / (c_img_max - c_img_min) - 1
        # 通道重排：将最大通道值替换为对应的 c_b, c_g, c_r
        # 计算每个通道的平均值
        # c_mean_values = torch.mean(c_img_normalized, dim=(0, 2, 3))  # 计算B, G, R通道的平均值
        # # 获取通道的排序索引，按平均值从大到小排序
        # _, sorted_indices = torch.sort(c_mean_values, descending=True)
        # # 创建一个新的张量用于重排
        # c_img_reordered = torch.zeros_like(c_img_normalized)
        # # 根据排序后的索引重排通道
        # for i in range(c_img_reordered.shape[0]):
        #     for j in range(3):
        #         c_img_reordered[i, j] = c_img_normalized[i, sorted_indices[j]]
        #
        # c_img = c_img_reordered
        return c_img

    def is_uniform(self, image):
        """检查图像是否为纯色"""
        img_np = image[0].cpu().numpy()
        return np.std(img_np) < 1e-5  # 标准差非常小表示颜色一致

    def otsu_image(self, images, c_img, save_path='save'):
        B, C, H, W = images.shape
        # 初始化 masks 和 mask_reverse
        masks = torch.ones((B, H, W), dtype=torch.uint8, device=images.device)  # 背景 1, 前景 0
        mask_reverse = torch.zeros((B, H, W), dtype=torch.uint8, device=images.device)  # 背景 0, 前景 1

        for b in range(B):
            img_np = images[b, 0].cpu().numpy()  # 将图像转换为 NumPy 数组

            # 检查当前图像是否为纯色
            if self.is_uniform(images[b]):
                # 如果是纯色图像，直接返回原图
                return images, c_img, masks, mask_reverse  # 返回原图

            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # 归一化
            threshold = filters.threshold_otsu(img_np)  # 使用 skimage 计算 Otsu 阈值
            # threshold = threshold_sauvola(gray_image, window_size=16)  # 使用 skimage 计算 Otsu 阈值
            mask_np = (img_np > threshold).astype(np.uint8)  # 创建二值掩膜，前景为0，背景为1
            mask_np_reverse = 1 - mask_np  # 反转掩膜，前景为1，背景为0

            masks[b] = torch.from_numpy(mask_np).to(masks.device)  # 更新 masks
            mask_reverse[b] = torch.from_numpy(mask_np_reverse).to(masks.device)  # 更新 mask_reverse

        # 将掩膜应用于原图
        result_x = images * mask_reverse.unsqueeze(1)  # 使用 mask_reverse 应用于原图
        result = c_img * mask_reverse.unsqueeze(1)  # 使用 mask_reverse 应用于另一个模态

        return result_x, result, masks, mask_reverse

    # def otsu_image(self, image):
    #     c_img = self.IIC_sort(image)
    #     otsu_thresholds, masks = self.otsu(image)
    #
    #     # 将掩膜归一化为0和1
    #     masks = masks / 255  # 将255的掩膜转换为1
    #
    #     # 将掩膜应用于原图
    #     result = c_img * masks.unsqueeze(1)  # 广播到 [B, 1, H, W]
    #
    #     return result


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
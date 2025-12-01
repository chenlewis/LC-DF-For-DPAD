# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import Sequential

from ..builder import HEADS
from .cls_head import ClsHead
import math
from .mlp import MLP
# from torch import nn
from typing import List, Type
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

@HEADS.register_module()
class SwinHead(ClsHead):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to ``dict(type='Tanh')``.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=None,
                 mlp_num = 1,
                 mlp_act = nn.GELU,
                 act_cfg=dict(type='Tanh'),
                 init_cfg=dict(type='Constant', layer='Linear', val=0),
                 mlp_cfg = False,
                 *args,
                 **kwargs):
        super(SwinHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg
        self.mlp_cfg = mlp_cfg
        self.mlp_num = mlp_num
        self.mlp_act = mlp_act

        self.fusion_mlp = Mlp(in_features=2048, hidden_features=None, out_features=1024,act_layer=nn.ReLU, drop=0.)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        if self.mlp_cfg:

            layers = [
                ('head1', nn.Linear(1024, 512)),
                ('relu1', nn.ReLU(inplace=True)),
                ('head2', nn.Linear(512, 128)),
                ('relu2', nn.ReLU(inplace=True)),
                ('head', nn.Linear(128, 2))
            ]

        elif self.hidden_dim is None:

            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:

            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))
        print("layer",self.layers)
    def init_weights(self):
        super(VisionTransformerClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)


    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
            print("x.shape",x.shape)

        device = x.device
        if len(x.shape) == 2:  # 如果 x 是二维张量，直接返回
            cls_token = x
        else:  # 如果 x 是三维张量
            n = x.shape[1] // 49  # 这里的257是token数，可以根据你的模型调整
            num_tokens = x.shape[1] // n  # 假设拼接后 token 数是 2 倍的原始序列

            if n == 1:
                # 对第一个模态进行池化
                x_cls = x[:, 0, :]  # 第一个模态的 token
                cls_token = x_cls
            elif n == 2:
                # 分别对两个模态进行池化
                x_fm = x[:, :num_tokens, :]  # 第一个模态
                z_fm = x[:, num_tokens:, :]  # 第二个模态
                # 使用池化操作分别提取每个模态的全局表示
                x_pool = F.adaptive_avg_pool1d(x_fm.permute(0, 2, 1), 1).squeeze(-1)  # 池化到每个通道的平均值
                z_pool = F.adaptive_avg_pool1d(z_fm.permute(0, 2, 1), 1).squeeze(-1)  # 池化到每个通道的平均值
                # 将两个模态的池化结果合并
                cls_token = torch.cat([x_pool, z_pool], dim=1)  # [B, D * 2]

                # 可以加一个融合层（如MLP）进一步处理合并后的结果
                cls_token = self.fusion_mlp(cls_token)  # [B, D]

        if self.mlp_cfg:
            x = self.layers.head1(cls_token)
            # x = self.layers.head1(fused_cls_token)
            x = self.layers.relu1(x)
            x = self.layers.head2(x)
            x = self.layers.relu2(x)
            x = self.layers.head(x)
            # return self.layers.head(cls_token)
            return x
        if self.hidden_dim is None:
            return self.layers.head(cls_token)
        else:
            x = self.layers.pre_logits(cls_token)
            return self.layers.head(self.layers.act(x))


    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[tuple[tensor, tensor]]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. Every item should be a tuple which
                includes patch token and cls token. The cls token will be used
                to classify and the shape of it should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.layers.head(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):

        if x.dim() != 2:
            raise ValueError("Expected x to have shape [batch_size, num_classes]")
        num_classes = x.size(1)
        gt_label_one_hot = F.one_hot(gt_label.long(), num_classes=num_classes).float()
        gt_label = gt_label_one_hot.float()  # 如果gt_label最初是float类型

        losses = self.loss(x, gt_label, **kwargs)

        return losses

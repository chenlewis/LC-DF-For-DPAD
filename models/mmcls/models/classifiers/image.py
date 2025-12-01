# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier
import torch.nn.functional as F
import torch
import torch.nn as nn

@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            print("buidling head")
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def forward_dummy(self, img,img_FMAG):
        """Used for computing network flops.

        See `mmclassificaiton/tools/analysis_tools/get_flops.py`
        """
        output = self.extract_feat(img,img_FMAG, stage='pre_logits')
        pred = output
        softmax = True
        # if softmax:
        #     # pred = (
        #     #     F.softmax(output, dim=1) if output is not None else None)
        #     pred = torch.sigmoid(pred) if pred is not None else None
        # else:
        #     pred = output
        return pred
        # return self.extract_feat(img, stage='pre_logits')

    def extract_multiple_feats(self, img, img_FMAG=None):
        # Extract backbone features
        if img_FMAG is not None:
            backbone_features = self.backbone(img, img_FMAG)
        else:
            backbone_features = self.backbone(img)

        # Pass backbone features through neck
        if self.with_neck:
            neck_features = self.neck(backbone_features)
        else:
            neck_features = backbone_features

        # Get pre-logits
        if self.with_head and hasattr(self.head, 'pre_logits'):
            pre_logits_features = self.head.pre_logits(neck_features)
        else:
            pre_logits_features = None

        return {
            'backbone': backbone_features,
            'neck': neck_features,
            'pre_logits': pre_logits_features
        }

    def extract_feat(self, img,img_FMAG=None, stage='pre_logits'):
        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')
        # print("img:", img.shape)
        # x = self.backbone(img)
        if img_FMAG!=None:
            x = self.backbone(img,img_FMAG)
        else:
            x = self.backbone(img)
        # 如果使用 attention 信息对特征加权
        # 如果存在 attn 和 attn_i，将其拼接后用于加权
        # if "attn" in aux_dict and aux_dict["attn"] is not None and "attn_i" in aux_dict and aux_dict[
        #     "attn_i"] is not None:
        #     # 平均多头注意力并拼接
        #     attn_combined = torch.cat((aux_dict["attn"].mean(dim=1), aux_dict["attn_i"].mean(dim=1)), dim=1)
        #     # 加权拼接后的特征
        #     x = x * attn_combined.unsqueeze(-1)
        # print("X:",type(x), len(x) if isinstance(x, (list, tuple)) else 'Not iterable')
        # i=0
        if stage == 'backbone':
            return x
        # print("x",x[-1].shape)

        if self.with_neck:
            x = self.neck(x)
            # x = upmodel(x)
        if stage == 'neck':
            # x = upmodel(x)
            return x
        # print("x_after neck",x)
        if self.with_head and hasattr(self.head, 'pre_logits'):
            # print("with head")
            x = self.head.pre_logits(x)
        softmax = True
        pred = x
        return pred

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                should be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        # print("img",img)
        x = self.extract_feat(img)
        # print("x", x)
        losses = dict()
        loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, **kwargs)

        return res

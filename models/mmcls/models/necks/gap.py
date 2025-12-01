# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS


# @NECKS.register_module()
# class GlobalAveragePooling(nn.Module):
#     """Global Average Pooling neck.
#
#     Note that we use `view` to remove extra channel after pooling. We do not
#     use `squeeze` as it will also remove the batch dimension when the tensor
#     has a batch dimension of size 1, which can lead to unexpected errors.
#
#     Args:
#         dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
#             Default: 2
#     """
#
#     def __init__(self, dim=2):
#         super(GlobalAveragePooling, self).__init__()
#         assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
#             f'{1, 2, 3}, get {dim} instead.'
#         if dim == 1:
#             self.gap = nn.AdaptiveAvgPool1d(1)
#         elif dim == 2:
#             self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         else:
#             self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
#
#     def init_weights(self):
#         pass
#
#     def forward(self, inputs):
#         if isinstance(inputs, tuple):
#             outs = tuple([self.gap(x) for x in inputs])
#             outs = tuple(
#                 [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
#         elif isinstance(inputs, torch.Tensor):
#             outs = self.gap(inputs)
#             outs = outs.view(inputs.size(0), -1)
#         else:
#             raise TypeError('neck inputs should be tuple or torch.tensor')
#         return outs
@NECKS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2], 'GlobalAveragePooling dim only supports 1 and 2, ' \
            f'got {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            # For 3D inputs with shape [batch_size, sequence_length, hidden_dim]
            self.gap = nn.AdaptiveAvgPool1d(1)  # Apply 1D pooling to the sequence dimension

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x.transpose(1, 2)).transpose(1, 2) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            # Ensure the input tensor has shape [batch_size, sequence_length, hidden_dim]
            assert inputs.dim() == 3, 'Input tensor should be 3D with shape [batch_size, sequence_length, hidden_dim]'
            outs = self.gap(inputs.transpose(1, 2)).transpose(1, 2)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
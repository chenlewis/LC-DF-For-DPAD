import torch
from torch import nn
import timm
import math
import torch.nn.functional as F

'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_direct_adapter(nn.Module):
    def __init__(self, dim,scale=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(dim, scale)
        self.adapter_up = nn.Linear(scale, dim)
        # self.adapter_mid=nn.Sequential(nn.Linear(dim, dim),nn.GELU())
        self.adapter_mid = nn.Linear(scale, scale)
        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        # Access the Linear layer within adapter_mid
        # nn.init.zeros_(self.adapter_mid[0].weight)  # First layer in adapter_mid
        # nn.init.zeros_(self.adapter_mid[0].bias)  # First layer in adapter_mid
        # Note: GELU activation doesn't have weights or biases to initialize

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.01)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        residual = x
        x_down = self.adapter_down(x)   
        # x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        # x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        # return x_up+residual
        return x_up

class Single_adapter(nn.Module):
    def __init__(self, embed_dim,scale_factor=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(embed_dim, embed_dim//scale_factor)
        # self.adapter_up = nn.Linear(768//scale_factor, 768)
        self.adapter_mid=nn.Sequential(
            nn.Linear(embed_dim//scale_factor, embed_dim//scale_factor),
            nn.GELU())
        # self.adapter_mid = nn.Linear(dim, dim)
        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        # Access the Linear layer within adapter_mid
        nn.init.zeros_(self.adapter_mid[0].weight)  # First layer in adapter_mid
        nn.init.zeros_(self.adapter_mid[0].bias)  # First layer in adapter_mid
        # Note: GELU activation doesn't have weights or biases to initialize

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.weight)
        # nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.01)
        self.dim = embed_dim//scale_factor

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        return x_down
        # return x_up


class Single_adapter_s(nn.Module):
    def __init__(self, embed_dim,scale_factor=8, xavier_init=False):
        super().__init__()
        # print("embed_dim",embed_dim)
        self.adapter_down = nn.Linear(embed_dim, embed_dim//scale_factor)
        self.adapter_up = nn.Linear(embed_dim//scale_factor, embed_dim)
        self.adapter_mid=nn.Sequential(
            nn.Linear(embed_dim//scale_factor, embed_dim//scale_factor),
            nn.GELU())
        # self.adapter_mid = nn.Linear(dim, dim)
        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        # Access the Linear layer within adapter_mid
        nn.init.zeros_(self.adapter_mid[0].weight)  # First layer in adapter_mid
        nn.init.zeros_(self.adapter_mid[0].bias)  # First layer in adapter_mid
        # Note: GELU activation doesn't have weights or biases to initialize

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.weight)
        # nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = embed_dim//scale_factor

    def forward(self, x):
        B, N, C = x.shape
        # print(f"Input x shape: {x.shape}")  # Debugging: Print input shape
        x_down = self.adapter_down(x)
        # print(f"After adapter_down x shape: {x_down.shape}")  # Debugging: Print shape after adapter_down
        x_mid = self.adapter_mid(x_down)
        # print(f"After adapter_mid x shape: {x_mid.shape}")  # Debugging: Print shape after adapter_mid
        x_up = self.adapter_up(x_mid)
        # print(f"After adapter_up x shape: {x_up.shape}")  # Debugging: Print shape after adapter_up
        x_up = self.dropout(x_up)
        return x_up

# class MonaAdapter(nn.Module):
#     def __init__(self, in_dim=768, scale_factor=8, dropout_rate=0.1):
#         super().__init__()
#         # 降维
#         self.down_dim = in_dim // scale_factor
#         self.adapter_down = nn.Linear(in_dim, self.down_dim)
#
#         # MonaOp: 使用深度可分离卷积提取多尺度特征
#         self.conv3 = nn.Conv2d(self.down_dim, self.down_dim, kernel_size=3, padding=1, groups=self.down_dim)
#         self.conv5 = nn.Conv2d(self.down_dim, self.down_dim, kernel_size=5, padding=2, groups=self.down_dim)
#         self.conv7 = nn.Conv2d(self.down_dim, self.down_dim, kernel_size=7, padding=3, groups=self.down_dim)
#         # self.conv11 = nn.Conv2d(self.down_dim, self.down_dim, kernel_size=11, padding=5, groups=self.down_dim)
#
#         # 线性投影回原始维度
#         self.adapter_up = nn.Linear(self.down_dim, in_dim)
#
#         # Dropout
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, x, hw_shape):
#         """
#         Args:
#             x: 输入特征 (B, N, C)，其中 B 是批量大小，N 是序列长度，C 是特征维度。
#             hw_shape: 输入特征的空间维度 (H, W)。
#         """
#         identity = x  # 保存输入特征
#         B, N, C = x.shape
#         H, W = hw_shape
#
#         # 降维
#         x_down = self.adapter_down(x)
#
#         # 转换为卷积需要的 (B, C, H, W) 格式
#         x_down = x_down.view(B, H, W, -1).permute(0, 3, 1, 2)
#
#         # 多尺度卷积
#         conv3_out = self.conv3(x_down)
#         conv5_out = self.conv5(x_down)
#         conv7_out = self.conv7(x_down)
#         # conv16_out = self.conv11(x_down)
#
#         # 多尺度特征融合
#         x_fused = (conv3_out + conv5_out + conv7_out) / 3.0 + x_down  # 残差连接
#         # x_fused = (conv3_out + conv5_out + conv16_out) / 3.0 + x_down  # 残差连接
#         # 转回 (B, N, C) 格式
#         x_fused = x_fused.permute(0, 2, 3, 1).reshape(B, N, -1)
#
#         # 投影回原始维度
#         x_up = self.adapter_up(x_fused)
#         x_out = self.dropout(x_up)
#
#         return identity + x_out  # 残差连接
class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        # x = (conv1_x + conv2_x + conv3_x) / 3.0

        identity = x

        x = self.projector(x)

        return identity + x
        # return x
class MonaAdapter0(nn.Module):
    def __init__(self, in_dim=768, scale_factor=8, dropout_rate=0.1):
        super().__init__()
        # 降维
        self.down_dim = in_dim // scale_factor
        self.adapter_down = nn.Linear(in_dim, self.down_dim)
        # MonaOp: 使用更高级的 MonaOp 模块提取多尺度特征
        self.mona_op = MonaOp(self.down_dim)  # MonaOp 替换了原先的单独卷积操作
        # 线性投影回原始维度
        self.adapter_up = nn.Linear(self.down_dim, in_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # 增加动态调整的可学习参数（类似 Mona 中的 gamma 和 gammax）
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        # 添加 LayerNorm 提升正则化能力
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, hw_shape):
        """
        Args:
            x: 输入特征 (B, N, C)，其中 B 是批量大小，N 是序列长度，C 是特征维度。
            hw_shape: 输入特征的空间维度 (H, W)。
        """
        identity = x  # 保存输入特征
        B, N, C = x.shape
        H, W = hw_shape

        # 动态特征调整（增加 Mona 的正则化方式）
        x = self.norm(x) * self.gamma + x * self.gammax
        # 降维
        x_down = self.adapter_down(x)
        # 转换为卷积需要的 (B, C, H, W) 格式
        x_down = x_down.view(B, H, W, -1).permute(0, 3, 1, 2)
        # 使用 MonaOp 提取多尺度特征
        x_fused = self.mona_op(x_down)  # MonaOp 内部已实现多尺度卷积和融合
        # 转回 (B, N, C) 格式
        x_fused = x_fused.permute(0, 2, 3, 1).reshape(B, N, -1)
        # 投影回原始维度
        x_up = self.adapter_up(x_fused)
        x_out = self.dropout(x_up)
        return identity + x_out  # 残差连接

class MonaAdapter(nn.Module):
    def __init__(self, in_dim=768, scale_factor=8, dropout_rate=0.1):
        super().__init__()
        # 降维
        self.down_dim = in_dim // scale_factor
        self.adapter_down = nn.Linear(in_dim, self.down_dim)
        # MonaOp: 使用更高级的 MonaOp 模块提取多尺度特征
        self.adapter_conv = MonaOp(self.down_dim)  # MonaOp 替换了原先的单独卷积操作
        # 线性投影回原始维度
        self.adapter_up = nn.Linear(self.down_dim, in_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # 增加动态调整的可学习参数（类似 Mona 中的 gamma 和 gammax）
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        # 添加 LayerNorm 提升正则化能力
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, hw_shape):
        """
        Args:
            x: 输入特征 (B, N, C)，其中 B 是批量大小，N 是序列长度，C 是特征维度。
            hw_shape: 输入特征的空间维度 (H, W)。
        """
        # identity = x  # 保存输入特征
        B, N, C = x.shape
        H, W = hw_shape
        # 动态特征调整（增加 Mona 的正则化方式）
        x = self.norm(x) * self.gamma + x * self.gammax
        identity = x  # 保存输入特征
        # 降维
        project1 = self.adapter_down(x)
        # 转换为卷积需要的 (B, C, H, W) 格式
        project1 = project1.view(B, H, W, -1).permute(0, 3, 1, 2)
        # 使用 MonaOp 提取多尺度特征
        project1 = self.adapter_conv(project1)  # MonaOp 内部已实现多尺度卷积和融合
        # 转回 (B, N, C) 格式
        project1 = project1.permute(0, 2, 3, 1).reshape(B, N, -1)
        # 非线性激活
        nonlinear = F.gelu(project1)
        # Dropout
        nonlinear = self.dropout(nonlinear)
        # 投影回原始维度
        project2 = self.adapter_up(nonlinear)
        return identity + project2  # 残差连接
        # return project2

"""


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        #print(x.shape)
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        #print(x_down.shape)

        x_patch = x_down[:, 64:].reshape(B, 16, 16, self.dim).permute(0, 3, 1, 2)   ############
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 16 * 16, self.dim)


        #x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up
"""
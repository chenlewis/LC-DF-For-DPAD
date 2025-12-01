import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
                 
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            # print("mask.shape", mask.shape)
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
            # print("attn.shape",attn.shape)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print("x.shape", x.shape)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14, window_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        B, N, C = x.shape  # B: batch size, N: number of tokens (e.g., patches), C: channels/features

        # Linear projection of input into Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Compute the attention logits (dot product of Q and K)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply relative position bias if specified
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        # Apply mask to attention logits
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Sparse attention: only focus on local window
        # Create a local window mask for sparse attention
        if self.window_size > 1:
            attn = self.sparse_window_attention(attn)

        # Softmax and apply dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute output: weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x

    def sparse_window_attention(self, attn):
        """Apply sparse attention by limiting attention to a local window."""
        B, H, N, _ = attn.shape
        half_window = self.window_size // 2

        # We use a sliding window approach to make the attention sparse
        for b in range(B):
            for h in range(H):
                for i in range(N):
                    # Get the current token's position
                    start_idx = max(0, i - half_window)
                    end_idx = min(N, i + half_window + 1)

                    # Set all other positions outside the local window to -inf
                    attn[b, h, i, :start_idx] = float('-inf')
                    attn[b, h, i, end_idx:] = float('-inf')

        return attn

class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x







class Cross_Attention(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, dom, ref, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = dom.shape
        
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        """
        elif self.mode=='t2t':  # Template to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_z, C
            v = x[:, lens_z:]  # B, lens_z, C
        elif self.mode=='s2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C
        """

        """
        q = dom[:, :lens_x]  # B, lens_x, C
        k = ref[:, lens_x:]  # B, lens_x, C
        v = ref[:, lens_x:]  # B, lens_x, C
        """

        q = dom
        k = ref
        v = ref

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z
       
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C

        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        """
        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)
        """

        if return_attention:
            return x, attn
        else:
            return x
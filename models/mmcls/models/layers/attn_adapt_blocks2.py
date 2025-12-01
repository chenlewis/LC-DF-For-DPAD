import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from .attn import Attention
from .adapter import Bi_direct_adapter, Single_adapter, MonaAdapter

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .deit import selector, Attention_cfged, Mlp_cfged

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float,
                          global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        # print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)

    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    # print("finish ce func")

    return tokens_new, keep_index, removed_index  # x, global_index_search, removed_index_search


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

        # print("z.shape",z.shape)
        # lens_t = global_index_template.shape[1]

        # xori = x
        # x_attn, attn = self.attn(self.norm1(x), mask, True)
        # x = x + self.drop_path(x_attn) + self.drop_path(
        #     self.adap_t(self.norm1(z)))  #########-------------------------adapter
        #
        # # xi_attn, i_attn = self.attn(self.norm1(z), mask, True)
        # # z = z + self.drop_path(xi_attn) + self.drop_path(
        # #     self.adap_t(self.norm1(xori)))
        # zi_attn, i_attn = self.attn(self.norm1(z), mask, True)
        # # i_attn = attn
        # z = z+ self.drop_path(zi_attn) + self.drop_path(
        #     self.adap_t(self.norm1(xori)))
        # #########-------------------------adapter
        #
        # # lens_t = global_index_template.shape[1]
        # # removed_index_search = None
        # # removed_index_searchi = None
        # # if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
        # #     keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
        # #     x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search,
        # #                                                                          global_index_search, ce_template_mask)
        # #     xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t,
        # #                                                                             keep_ratio_search,
        # #                                                                             global_index_searchi,
        # #                                                                             ce_template_mask)
        #
        # xori = x
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(
        #     self.adap2_t(self.norm2(z)))+self.drop_path(adapter_up(self.adap_st(self.norm1(x_origin))))  ###-------adapter
        #
        # z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(
        #     self.adap2_t(self.norm2(xori)))+self.drop_path(adapter_up(self.adap_st_p(self.norm1(x_origin))))  ###-------adapter

        # return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn
        return x, attn, z, i_attn


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
        self.adap_st = Single_adapter(scale_factor)
        # self.adap_st_p = Single_adapter(scale_factor)
        self.adap_t = Bi_direct_adapter(adapter_dim)
        # self.adap_t_p = Bi_direct_adapter(adapter_dim)
        self.adap2_t = Bi_direct_adapter(adapter_dim)

    def forward(self, x_otsu, z_otsu, x_FMAG, x, z, x_origin, z_origin, adapter_up, adapter_up2, i,
                global_index_template,
                global_index_templatei, global_index_search, global_index_searchi, mask_x=None,
                mask_z=None, ce_template_mask=None, keep_ratio_search=None):

        xor = x
        zor = z
        # print("xi", xi.shape)
        # 更新 z 通过 DropPath 和适配器

        mask_x = None
        mask_z = None
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
            # x = x + self.drop_path(x_attn)+ self.drop_path(self.adap_t(self.norm1(zor))) #########-------------------------adapter
            x = x + self.drop_path(x_attn)
            # lens_z_new = global_index_template.shape[1]
            z_attn, i_attn = self.attn(self.norm1(z), mask_z, True)
            z = z + self.drop_path(z_attn)
            # z = z + self.drop_path(z_attn)+ self.drop_path(self.adap_t(self.norm1(xor))) #########-------------------------adapter
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
            # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(zor)))
            # lens_z_new = global_index_template.shape[1]
            z_attn, i_attn = self.attn(self.norm1(z), mask_z, True)
            z = z + self.drop_path(z_attn)
            # z = z + self.drop_path(z_attn) + self.drop_path(self.adap_t(self.norm1(xor)))
            xori = x
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))

        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))
        z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(self.adap2_t(self.norm2(xori)))

        # z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(self.adap2_t(self.norm2(xori)))+self.drop_path(self.adap_st_p(self.norm2(z_origin)))
        # z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(self.adap2_t(self.norm2(xori)))+self.drop_path(adapter_up2(self.adap_st_p(self.norm2(z_origin))))
        # z = x
        # i_attn = attn
        # x = torch.cat([x_FMAG, x], dim=1)
        return x, attn, z, i_attn


class CEABlock_FMAG0(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, adapter_dim=16,
                 scale_factor=8, n_tokens=256, cutoff=0.15):
        super().__init__()
        self.n_tokens = n_tokens
        self.cutoff = cutoff

        # Norm and Attention (using Attention_cfged)
        self.norm1 = norm_layer(dim)
        self.attn = Attention_cfged(
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            dim_msa=[[dim, dim], [dim, dim], [dim, dim], [dim, dim]],
            n_token=n_tokens
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Norm and MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_cfged(
            in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop
        )

        # Adapter layers
        self.adap_st = Single_adapter(scale_factor)
        self.adap_st_p = Single_adapter(scale_factor)
        self.adap_t = Bi_direct_adapter(adapter_dim)
        self.adap2_t = Bi_direct_adapter(adapter_dim)

        # Gamma parameters
        self.gamma_1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x_otsu, z_otsu, x_FMAG, x, z, x_origin, z_origin, adapter_up, adapter_up2, i,
                global_index_template, global_index_templatei, global_index_search, global_index_searchi,
                mask_x=None, mask_z=None, ce_template_mask=None, keep_ratio_search=None):
        xor = x  # 保存原始 x
        zor = z  # 保存原始 z
        # Step 1: 更新 z 的原始值
        z_origin = adapter_up2(self.adap_st_p(self.norm2(z_origin)))

        # Step 2: x_FMAG 的处理
        if x_FMAG is not None:
            x_FMAG = adapter_up(self.adap_st(self.norm2(x_FMAG)))
            x = torch.cat((x[:, :1, :], x_FMAG + x[:, 1:, :]), dim=1)

        # Step 3: Attention 和 selector
        x_msa, attn = self.attn(self.norm1(x))
        x = self.drop_path(self.gamma_1 * x_msa) + x  # MSA 加入残差连接
        x = selector(x, attn, self.n_tokens, cutoff=self.cutoff)  # 应用 selector 对 x 的 token 进行选择

        # Step 4: 更新 z 的 Attention
        z_msa, i_attn = self.attn(self.norm1(z))
        z = self.drop_path(self.gamma_1 * z_msa) + z  # MSA 加入残差连接
        z = selector(z, i_attn, self.n_tokens, cutoff=self.cutoff)  # 应用 selector 对 x 的 token 进行选择
        # z = z + self.drop_path(z_attn)

        # Step 5: 使用 Adapter 融合 x 和 z
        xori = x
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))

        z = z + self.drop_path(self.gamma_2 * self.mlp(self.norm2(z))) + self.drop_path(self.adap2_t(self.norm2(xori)))

        return x, attn, z, i_attn


class CEABlock_FMAG1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                 adapter_dim=16, scale_factor=8, kernel_size=7, freq_dropout_rate=0.5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Adapter 和其他模块
        self.adap_st = Single_adapter(scale_factor)
        self.adap_t = Bi_direct_adapter(adapter_dim)
        self.adap2_t = Bi_direct_adapter(adapter_dim)

        # 频域滤波器模块
        self.freq_filter = FrequencyFilter(kernel_size=7, dropout_rate=0.5, weights=[0.5, 0.5, 0., 0., 0.])

    def forward(self, x_otsu, z_otsu, x_FMAG, x, z, x_origin, z_origin, adapter_up, fusion_layer, i,
                global_index_template, global_index_templatei, global_index_search, global_index_searchi,
                mask_x=None, mask_z=None, ce_template_mask=None, keep_ratio_search=None):
        # 原始 x 和 z 特征的存储
        xor = x
        zor = z
        cls_token = x_origin[:, 0:1, :]
        cls_token2 = z_origin[:, 0:1, :]
        mask_x = None
        mask_z = None
        # 对 x 和 z 应用频域滤波器
        x = self.freq_filter(x)
        z = self.freq_filter(z)
        # 对 x 应用 Attention 和 DropPath
        x_attn, attn = self.attn(self.norm1(x), mask_x, True)
        x = x + self.drop_path(x_attn)

        # 对 z 应用 Attention 和 DropPath
        z_attn, i_attn = self.attn(self.norm1(z), mask_z, True)
        z = z + self.drop_path(z_attn)

        xori = x
        # 如果 x_FMAG 存在，则进行进一步的适配器更新
        if x_FMAG is not None:
            cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)  # 定义全 0 的 cls_token
            x_FMAG = torch.cat([cls_token, x_FMAG], dim=1)  # 拼接 cls_token 和 x_FMAG
            x_FMAG = self.freq_filter(x_FMAG)
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(
                self.adap2_t(self.norm2(z))) + self.drop_path(
                self.adap_st(self.norm1(x_FMAG)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(z)))

        # 对 z 的更新
        cls_token2 = torch.zeros(z.size(0), 1, z.size(2), device=x.device)
        z_origin = torch.cat([cls_token2, z_origin], dim=1)
        z = z + self.drop_path(self.mlp(self.norm2(z))) + self.drop_path(self.adap2_t(self.norm2(xori)))

        return x, attn, z, i_attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        # print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

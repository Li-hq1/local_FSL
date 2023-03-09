# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import random 

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
from models.nextvlad import NeXtVLAD, NeXtVLAD_Centers

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, pooling='mean', mask_ratio=.0, 
                 nextvlad_lamb=4, nextvlad_cluster=64, nextvlad_groups=8, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        # pooling preperation
        self.pooling = pooling
        if self.pooling != 'cls':
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            del self.norm  # remove the original norm

        # pooling
        if self.pooling == 'mean':
            self.fc_norm = norm_layer(embed_dim)
            self.act = nn.ReLU()
        elif self.pooling == 'nextvlad':
            num_patches = self.patch_embed.num_patches
            self.before_norm = norm_layer(embed_dim)
            self.after_norm = norm_layer(embed_dim)
            self.nextvlad = NeXtVLAD(
                feature_size=embed_dim,
                max_frames=num_patches,
                nextvlad_cluster_size=nextvlad_cluster,
                lamb=nextvlad_lamb,
                groups=nextvlad_groups,
            )
            nextvlad_dim = (nextvlad_cluster * nextvlad_lamb * embed_dim // nextvlad_groups)
            self.dim_reduce = nn.Linear(nextvlad_dim, embed_dim)
            self.act = nn.ReLU()
        elif self.pooling == 'nextvlad_centers':
            num_patches = self.patch_embed.num_patches
            self.before_norm = norm_layer(embed_dim)
            self.after_norm = norm_layer(embed_dim)
            self.cluster_weights = nn.Parameter(torch.rand(1, int((nextvlad_lamb * embed_dim) // nextvlad_groups), nextvlad_cluster))
            self.nextvlad = NeXtVLAD_Centers(
                feature_size=embed_dim,
                max_frames=num_patches,
                nextvlad_cluster_size=nextvlad_cluster,
                lamb=nextvlad_lamb,
                groups=nextvlad_groups,
                cluster_weights=self.cluster_weights
            )
            nextvlad_dim = (nextvlad_cluster * nextvlad_lamb * embed_dim // nextvlad_groups)
            self.dim_reduce = nn.Linear(nextvlad_dim, embed_dim)
            self.act = nn.ReLU()
            
            # torch.nn.init.normal_(self.cluster_weights, std=.02)
            # torch.nn.init.xavier_uniform_(self.cluster_weights.weight)
        
        trunc_normal_(self.head.weight, std=2e-5)
        
    def random_masking(self, x, mask_ratio, keep_pred):
        B, L, D = x.size()
        len_keep = int(L * (1 - mask_ratio))

        # noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # ids_restore = torch.argsort(ids_shuffle, dim=1) # [B, L]

        # kept patch selection
        ids_keep = torch.multinomial(keep_pred, num_samples=len_keep, replacement=False)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # mask input
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is remove, 1 is keep, opposite with pretrain
        mask = torch.zeros([B, L], device=x.device)
        mask[:, :len_keep] = 1
        # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        # return x_masked, mask, ids_restore
        return x_masked, mask
    
    def forward_features(self, x, keep_pred=None):
        # embed patches
        x = self.patch_embed(x)
        B, L, D = x.size()

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # masking
        mask_ratio = self.mask_ratio if self.training else 0
        # mask_ratio = random.uniform(max(0, self.mask_ratio-0.1), min(1, self.mask_ratio+0.1)) if self.training else 0
        if mask_ratio == 0:
            mask = torch.ones([B, L], device=x.device)
        else:
            x, mask = self.random_masking(x, mask_ratio, keep_pred)

        # add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # dropout
        x = self.pos_drop(x)

        # forward
        for blk in self.blocks:
            x = blk(x)

        # pooling
        if self.pooling == 'mean':
            x = x[:, 1:, :].mean(dim=1)  # global pooling without cls token
            # outcome = self.fc_norm(x)
            x = self.fc_norm(x)
            outcome = self.act(x)
        elif self.pooling == 'cls':
            x = self.norm(x)
            outcome = x[:, 0]
        elif self.pooling == 'nextvlad':
            # append zero tokens to sequence
            zero_tokens = torch.zeros([B, mask.shape[1] + 1 - x.shape[1], D], device=x.device, dtype=x.dtype)
            x = torch.cat([x[:, 1:, :], zero_tokens], dim=1)  # no cls token
            # x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle [B, L, D]
            # nextvlad
            x = self.before_norm(x)
            x = self.nextvlad(x, mask)
            x = self.dim_reduce(x)
            x = self.after_norm(x)
            # x -= 1.0
            outcome = self.act(x)
            return outcome
        # elif self.pooling == "nextvlad_centers":
        #     # append zero tokens to sequence
        #     zero_tokens = torch.zeros([B, ids_restore.shape[1] + 1 - x.shape[1], D], device=x.device, dtype=x.dtype)
        #     x_ = torch.cat([x[:, 1:, :], zero_tokens], dim=1)  # no cls token
        #     x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle [B, L, D]
        #     # nextvlad
        #     x = self.before_norm(x)
        #     x = self.nextvlad(x, mask)
        #     x = self.dim_reduce(x)
        #     x = self.after_norm(x)
        #     outcome = self.act(x)
        #     # outcome = x
        #     return outcome
        else: 
            assert False, 'Wrong pooling value!'

        return outcome

    def forward(self, x, keep_pred=None):
        rep = self.forward_features(x, keep_pred)
        ce = self.head(rep)
        if self.pooling == 'nextvlad_centers':
            return ce, self.cluster_weights.squeeze(0).T
        return ce


def vit_suptiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
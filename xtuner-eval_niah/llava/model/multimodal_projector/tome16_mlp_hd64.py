# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import torch.nn as nn
from typing import Callable, Tuple
import torch.nn.functional as F




def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int, 
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src) # , reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size




class ToMe16_mlp_hd64(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.mm_hidden_size = config.mm_hidden_size
        self.hw = vision_cfg.image_size // vision_cfg.patch_size
        self.num_attention_heads = vision_cfg.num_attention_heads
        self.mlp = nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size))
        self.max_pos_hw = self.hw
        self.max_pos_num_frames = config.mm_pos_num_frames
        # self._set_3d_pos_cache(max_grid_size=self.max_pos_hw, max_t_size=self.max_pos_num_frames)
        self.num_image_patches_per_side = 8
        self.num_frame_patches_per_side = 4
        
    def merge_tokens(self, x, target_num_token):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
                
        
        head = self.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2) # [b, p, c//head]
            merge, _ = bipartite_soft_matching(
                metric, 
                r
            )
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # x = x.reshape(-1, c)  # 300, 1024
        return x




    def forward(self, x, compress=False, local_num_frames=-1):

        height = width = self.hw
        assert height * width == x.shape[1]
        dtype = x.dtype
        device = x.device

        if local_num_frames != -1 and local_num_frames != 1:
            assert compress is True
        if compress:
            if local_num_frames != -1:
                num_frames = local_num_frames
                x = x.reshape(x.shape[0] // local_num_frames, -1, x.shape[-1])
            else:
                num_frames = x.shape[0]
                x = x.reshape(1, -1, x.shape[-1])
            num_tome_tokens = 16 * num_frames
        else:
            num_tome_tokens = 64
        
        x = self.merge_tokens(x, target_num_token=num_tome_tokens)
        x = self.mlp(x)

        return x

    @property
    def config(self):
        return {"mm_projector_type": "tome16_mlp_hd64"}


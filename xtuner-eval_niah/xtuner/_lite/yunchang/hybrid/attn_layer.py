from ..comm.all_to_all import SeqAllToAll4D, SeqAllToAll5D

import torch

from typing import Any
from torch import Tensor

import torch.distributed as dist
from .utils import RING_IMPL_DICT, RING_IMPL_QKVPACKED_DICT
from ..globals import PROCESS_GROUP
from ..ring import llama3_flash_attn_prepare_cu_seqlens, llama3_flash_attn_varlen_func
from xtuner._lite.parallel import all_to_all


class LongContextAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
    ) -> None:

        super(LongContextAttention, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        self.use_pack_qkv = use_pack_qkv
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            out = self.ring_attn_fn(
                qkv[0],
                qkv[1],
                qkv[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )
        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

            out = self.ring_attn_fn(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output


class LongContextAttentionQKVPacked(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
    ) -> None:

        super(LongContextAttentionQKVPacked, self).__init__()

        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        self.ring_attn_fn = RING_IMPL_QKVPACKED_DICT[ring_impl_type]

    def forward(
        self,
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # scatter 3, gather 1

        world_size = dist.get_world_size(self.ulysses_pg)

        if world_size > 1:
            qkv = SeqAllToAll5D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )

        out = self.ring_attn_fn(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
        )

        # print(f"out {out.shape}")

        if type(out) == tuple:
            out = out[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2

        if world_size > 1:
            out = SeqAllToAll4D.apply(
                self.ulysses_pg, out, self.gather_idx, self.scatter_idx - 1
            )
        # out e.g., [s/p::h]
        return out


def llama3_varlen_attention_sp_ulysses_ring(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Tensor,
        ulysses_pg,
        ring_pg,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        heads_k_stride=1,
):
    scatter_idx = 1
    gather_idx = 0

    ulysses_world_size = dist.get_world_size(ulysses_pg)
    if ulysses_world_size > 1:
        query = all_to_all(
            query, ulysses_pg, scatter_idx, gather_idx
        )
        key = all_to_all(
            key, ulysses_pg, scatter_idx, gather_idx
        )
        value = all_to_all(
            value, ulysses_pg, scatter_idx, gather_idx
        )

    ring_world_size = dist.get_world_size(ring_pg)

    ring_rank = dist.get_rank(ring_pg)
    (
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, causal, ring_rank, ring_world_size)

    out = llama3_flash_attn_varlen_func(
        query,
        key,
        value,
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride=key.shape[1] if heads_k_stride == -1 else heads_k_stride,
        local_k_slice=local_k_slice,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
        group=ring_pg
    )

    if type(out) == tuple:
        context_layer, _, _ = out
    else:
        context_layer = out

    if ulysses_world_size > 1:
        output = all_to_all(
            context_layer, ulysses_pg, gather_idx, scatter_idx
        )
    else:
        output = context_layer
    # out e.g., [s/p::h]
    return output


def attention_sp_ulysses_ring(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        ulysses_pg,
        ring_pg,
        ring_impl_type: str = "basic",
):
    scatter_idx = 2
    gather_idx = 1
    ulysses_world_size = dist.get_world_size(ulysses_pg)
    if ulysses_world_size > 1:
        query = SeqAllToAll4D.apply(
            ulysses_pg, query, scatter_idx, gather_idx
        )
        key = SeqAllToAll4D.apply(
            ulysses_pg, key, scatter_idx, gather_idx
        )
        value = SeqAllToAll4D.apply(
            ulysses_pg, value, scatter_idx, gather_idx
        )

    ring_attn_fn = RING_IMPL_DICT[ring_impl_type]
    out = ring_attn_fn(
        query,
        key,
        value,
        causal=True,
        group=ring_pg,
    )

    if type(out) == tuple:
        context_layer, _, _ = out
    else:
        context_layer = out

    if ulysses_world_size > 1:
        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            ulysses_pg, context_layer, gather_idx, scatter_idx
        )
    else:
        output = context_layer

    return output


class LongContextVarLenAttentionForLlaMa3(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
            self,
            scatter_idx: int = 2,
            gather_idx: int = 1
    ) -> None:
        super().__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        assert (
                self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            cu_seqlens: Tensor,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer (l,h,d)
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        query_layer = SeqAllToAll4D.apply(
            self.ulysses_pg, query[None], self.scatter_idx, self.gather_idx
        )
        key_layer = SeqAllToAll4D.apply(
            self.ulysses_pg, key[None], self.scatter_idx, self.gather_idx
        )
        value_layer = SeqAllToAll4D.apply(
            self.ulysses_pg, value[None], self.scatter_idx, self.gather_idx
        )

        ring_rank = dist.get_rank(self.ring_pg)
        ring_world_size = dist.get_world_size(self.ring_pg)
        (
            local_cu_seqlens_q,
            local_cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, causal, ring_rank, ring_world_size)

        out = llama3_flash_attn_varlen_func(
            query_layer[0],
            key_layer[0],
            value_layer[0],
            local_cu_seqlens_q,
            local_cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride=1,
            local_k_slice=local_k_slice,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer[None], self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output[0]

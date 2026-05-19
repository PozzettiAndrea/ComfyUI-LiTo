#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements spoint encoder, a test bed for investigating
# voxel and knn based attention.

import traceback
import typing as T

from lito.models._comfy_mlp import Mlp

from lito.models.layers import SwiGLU as _SwiGLU

import pytorch3d.ops
import torch

from lito.models.layers import FourierEmbed, PluckerEmbed, PointwiseResnet, RMSNorm
from lito.models.point_encoder import ShapeLatent
from lito.script_utils import config_utils
from plibs import ppoint
from contextlib import nullcontext as _nullcontext
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS


try:
    from third_party.TRELLIS.trellis.models.structured_latent_vae.base import SparseTransformerBase
    import third_party.TRELLIS.trellis.modules.sparse as sp
    from third_party.TRELLIS.trellis.modules.utils import convert_module_to_f16, convert_module_to_f32

    TRELLIS_IMPORTED = True
except:
    print("trellis not imported, need to do bash environment/setup_trellis.sh")
    print(f"\n{__file__=}")
    traceback.print_exc()
    print("\n\n")
    TRELLIS_IMPORTED = False


class PointwiseResnetVoxelBlock(torch.nn.Module):
    """
    The block contains multiple layers of pointwise resnet layers,
    followed by a knn downsampling.
    """

    def __init__(
        self,
        dim_coord: int,
        dim_in: int,  # after positional encoding
        dim_out: int,
        dim_hidden: int,
        num_layers: int,
        cell_width: float,
        shift_ratio: float = 0,  # unit: cell_width
        activation_fn: str = "swiglu",
        bias: bool = True,
        avg_coord: bool = True,  # during voxel downsampling, whether to use the average
        # of the coordinates within the voxel as the output coordinate or randomly select one
    ):
        super().__init__()

        self.dim_coord = dim_coord
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.avg_coord = avg_coord
        self.cell_width = cell_width
        self.shift_ratio = shift_ratio

        self.linear_in_coord = _ops().Linear(self.dim_coord, self.dim_hidden)
        self.linear_in_feature = _ops().Linear(self.dim_in, self.dim_hidden)
        blocks = []
        for i in range(num_layers):
            block = PointwiseResnet(
                dim_in=2 * self.dim_hidden,
                dim_out=self.dim_hidden,
                bias=bias,
                activation_fn=activation_fn,
                add_init_activation=True,
            )
            blocks.append(block)
        self.blocks = torch.nn.ModuleList(blocks)
        self.linear_out = _ops().Linear(self.dim_hidden, self.dim_out)
        self._init_parameteres()

    def _init_parameteres(self):
        if self.linear_in_coord is not None:
            torch.nn.init.xavier_uniform_(self.linear_in_coord.weight)
            if self.linear_in_coord.bias is not None:
                torch.nn.init.constant_(self.linear_in_coord.bias, 0)
        if self.linear_in_feature is not None:
            torch.nn.init.xavier_uniform_(self.linear_in_feature.weight)
            if self.linear_in_feature.bias is not None:
                torch.nn.init.constant_(self.linear_in_feature.bias, 0)
        if self.linear_out is not None:
            torch.nn.init.xavier_uniform_(self.linear_out.weight)
            if self.linear_out.bias is not None:
                torch.nn.init.constant_(self.linear_out.bias, 0)

    def _avg_and_spread(
        self,
        feature: torch.Tensor,  # (bn, d)
        linear_idx: torch.Tensor,  # (bn,) long
        total_cells: int,
    ):
        """
        average features in the same cluster and spread the averaged feature to the original location.

        Args:
            feature:
                (n1+n2+...+nb, d)  packed
            linear_idx:
                (n1+n2+...+nb,) long, cell index unique to bijk, from [0, total_cells-1]

        Returns:
            (bn, d)
        """
        bn, d = feature.shape

        # gather local region feature by averaging features within the same cluster
        feat_mean = torch.zeros(total_cells, d, dtype=feature.dtype, device=feature.device)  # (total_cells, d)
        feat_mean.scatter_reduce_(
            dim=0,
            index=linear_idx.unsqueeze(-1).expand(bn, d),  # (bn, d)
            src=feature,  # (bn, d)
            reduce="mean",
            include_self=False,  # important, do not want to include 0 and the count
        )  # (total_cells, d)

        # spread the averaged feature back to original locations
        feature = torch.gather(
            input=feat_mean,  # (total_cells, d)
            dim=0,
            index=linear_idx.unsqueeze(-1).expand(bn, d),  # (bn, d)
        )  # (bn, d)
        return feature  # (bn, d)

    def _forward(
        self,
        coord: ppoint.PackedPoint,
        feature: torch.Tensor,
        bijk_info: T.Dict[str, T.Any],
    ):
        """
        Args:
            coord:
                packed point (bn, dn)
            feature:
                (bn, dim_in)
            bijk_info:
                linear_idx:
                    (n1+n2+...+nb,) long, cell index unique to bijk, from [0, total_cells-1]
                new_seq_lens:
                    (b,) number of occupied cells for each sample in the batch
                cell_counts:
                    (total_cells,) long, number of points in each cell (corresponding to linear_idx)
                total_cells:
                    int, total number of cells
                forward_idxs:
                    (n1+n2+...+nb,), index to sort the points before performing block attention
                backward_idxs:
                    (n1+n2+...+nb,), index to sort the points back after performing block attention
                attn_biases:
                    (num_chunks,) list of attn_bias uses by xformer
                chunk_start_idxs:
                    (num_chunks+1,) index of sorted coord (ie, into 0..n1+n2+...+nb)

        Returns:
            coord:
                packed point (bn, dn)
            feature:
                (bn, dim_out)
        """
        bn, d = feature.shape
        dn = coord.dn
        total_cells = bijk_info["total_cells"]  # int
        linear_idx = bijk_info["linear_idx"]  # (bn,)
        cell_counts = bijk_info["cell_counts"]  # (total_cells,)
        forward_idxs = bijk_info["forward_idxs"]  # (bn,)
        new_seq_lens = bijk_info["new_seq_lens"]  # (b,)

        # sort by voxel
        _coord = coord.coord[forward_idxs]  # (bn, dn)  sorted by voxel
        feature = feature[forward_idxs]  # (bn, d)
        linear_idx = linear_idx[forward_idxs]  # (bn,)

        if self.avg_coord:
            # average xyz to get new xyz (so implicitly weighted by sample density)
            _new_coord = torch.zeros(total_cells, dn, dtype=coord.dtype, device=coord.device)  # (total_cells, dn)
            _new_coord.scatter_reduce_(
                dim=0,
                index=linear_idx.unsqueeze(-1).expand(bn, dn),  # (bn, dn) sorted
                src=_coord,  # (bn, dn) sorted
                reduce="mean",
                include_self=False,  # important, do not want to include 0 and the count
            )  # (total_cells, dn)
            new_coord = ppoint.PackedPoint(
                coord=_new_coord,  # (total_cells, dn)
                seq_lens=new_seq_lens,  # (b,)
                coord_lim=coord.coord_lim,  # (dn, 2)
            )  # (total_cells, dn)
        else:
            # randomly select a point in each voxel (grad checkpoint saves random state, so we do
            # not need to handle anything)
            idx_in_each_cell = torch.floor(
                torch.rand(total_cells, dtype=coord.dtype, device=coord.device)
                * cell_counts.to(dtype=coord.dtype, device=coord.device)
            ).long()  # (total_cells,), [0, cell_count[i]-1]
            ridxs = (
                torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long, device=coord.device),  # (1,)
                        cell_counts.cumsum()[:-1],  # (total_cells-1,)
                    ],
                    dim=0,
                )
                + idx_in_each_cell
            )  # (total_cells,)
            new_coord = ppoint.PackedPoint(
                coord=_coord[ridxs],  # (total_cells, dn)
                seq_lens=new_seq_lens,  # (b,)
                coord_lim=coord.coord_lim,
            )  # (total_cells, dn)

        x = torch.cat(
            [
                self.linear_in_coord(_coord),  # (bn, h)
                self.linear_in_feature(feature),  # (bn, h)
            ],
            dim=-1,
        )  # (bn, 2h)

        x = self.blocks[0](x)  # (bn, h)
        for block in self.blocks[1:]:
            x_pooled = self._avg_and_spread(feature=x, linear_idx=linear_idx, total_cells=total_cells)  # (bn, h)
            x = torch.cat([x, x_pooled], dim=-1)  # (bn, 2h)
            x = block(x)  # (bn, h)

        # since we use mean reduce, (avg first or avg later is the same)
        # ie, A (avg(f_i)) + b = avg(Afi + b)
        if self.dim_out < x.size(-1):
            x = self.linear_out(x)  # (bn, dim_out)
            # average feature within the same cluster
            new_feature = torch.zeros(total_cells, x.size(-1), dtype=x.dtype, device=x.device)  # (total_cells, d)
            new_feature.scatter_reduce_(
                dim=0,
                index=linear_idx.unsqueeze(-1).expand(bn, x.size(-1)),  # (bn, d)
                src=x,  # (bn, d)
                reduce="mean",
                include_self=False,  # important, do not want to include 0 and the count
            )  # (total_cells, d)
        else:
            # average feature within the same cluster
            new_feature = torch.zeros(total_cells, x.size(-1), dtype=x.dtype, device=x.device)  # (total_cells, h)
            new_feature.scatter_reduce_(
                dim=0,
                index=linear_idx.unsqueeze(-1).expand(bn, x.size(-1)),  # (bn, h)
                src=x,  # (bn, h)
                reduce="mean",
                include_self=False,  # important, do not want to include 0 and the count
            )  # (total_cells, h)
            new_feature = self.linear_out(new_feature)  # (total_cells, dim_out)

        return dict(
            coord=new_coord,  # packed points (total_cells, dn)
            feature=new_feature,  # (total_cells, dim_out)
        )

    def forward(
        self,
        coord: T.Union[torch.Tensor, ppoint.PackedPoint],
        feature: torch.Tensor,
        use_grad_checkpointing: bool = False,
    ):
        """
        Args:
            coord:
                (b, n, dn) or packed point (bn, dn)
            feature:
                (b, n, dim_in) or (bn, dim_in)

        Returns:
            coord:
                packed point (bm, dn)
            feature:
                (bm, dim_out)
        """

        if isinstance(coord, torch.Tensor):
            b, n, dn = coord.shape
            coord = ppoint.PackedPoint(
                coord=coord.reshape(b * n, dn),
                seq_lens=[n] * b,
            )
            feature = feature.reshape(b * n, dn)  # (bn, dn)
        elif isinstance(coord, ppoint.PackedPoint):
            pass
        else:
            raise NotImplementedError

        bn, d = feature.shape

        # determine voxel index
        bijk_info = coord.get_bijk_info(
            cell_width=self.cell_width,
            shift=self.shift_ratio * self.cell_width,
            save_to_cache=False,  # no need to save
        )

        if use_grad_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                coord,
                feature,
                bijk_info,
                use_reentrant=False,
            )
        else:
            return self._forward(
                coord=coord,
                feature=feature,
                bijk_info=bijk_info,
            )


class PointwiseResnetVoxelBlocks(torch.nn.Module):
    """
    A series of blocks containing multiple layers of pointwise resnet layers.
    """

    def __init__(
        self,
        dim_coord: int,
        dim_in: int,  # after positional encoding
        num_blocks: int,
        dim_outs: T.Union[int, T.List[int]],  # one for each block
        dim_hiddens: T.Union[int, T.List[int]],  # one for each block
        num_layers: T.Union[int, T.List[int]],  # one for each block
        cell_widths: T.Union[float, T.List[float]],  # one for each block
        shift_ratios: T.Union[float, T.List[float]] = 0,  # one for each block
        activation_fn: str = "swiglu",
        bias: bool = True,
        avg_coord: bool = True,  # during voxel downsampling, whether to use the average
        # of the coordinates within the voxel as the output coordinate or randomly select one
    ):
        super().__init__()

        if isinstance(dim_outs, int):
            dim_outs = [dim_outs] * num_blocks
        if isinstance(dim_hiddens, int):
            dim_hiddens = [dim_hiddens] * num_blocks
        if isinstance(num_layers, int):
            num_layers = [num_layers] * num_blocks
        if isinstance(cell_widths, (int, float)):
            cell_widths = [cell_widths] * num_blocks
        if isinstance(shift_ratios, (int, float)):
            shift_ratios = [shift_ratios] * num_blocks

        assert len(dim_outs) == num_blocks
        assert len(dim_hiddens) == num_blocks
        assert len(num_layers) == num_blocks
        assert len(cell_widths) == num_blocks
        assert len(shift_ratios) == num_blocks

        self.dim_coord = dim_coord
        self.dim_in = dim_in
        self.dim_outs = dim_outs
        self.dim_hiddens = dim_hiddens
        self.dim_out = dim_outs[-1]
        self.cell_widths = cell_widths
        self.shift_ratios = shift_ratios

        current_dim = dim_in
        blocks = []
        for i in range(num_blocks):
            block = PointwiseResnetVoxelBlock(
                dim_coord=dim_coord,
                dim_in=current_dim,
                dim_out=dim_outs[i],
                dim_hidden=dim_hiddens[i],
                num_layers=num_layers[i],
                cell_width=cell_widths[i],
                shift_ratio=shift_ratios[i],
                bias=bias,
                activation_fn=activation_fn,
                avg_coord=avg_coord,
            )
            blocks.append(block)
            current_dim = dim_outs[i]
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(
        self,
        coord: T.Union[torch.Tensor, ppoint.PackedPoint],
        feature: torch.Tensor,
        use_grad_checkpointing: bool = False,
    ):
        """
        Args:
            coord:
                (b, n, dn) or packed point (bn, dn)
            feature:
                (b, n, dim_in) or (bn, dim_in)

        Returns:
            coord:
                packed point (bm, dn)
            feature:
                (bm, dim_out)
        """

        for block in self.blocks:
            out_dict = block(
                coord=coord,
                feature=feature,
                use_grad_checkpointing=use_grad_checkpointing,
            )
            coord = out_dict["coord"]  # (bmi, dn)
            feature = out_dict["feature"]  # (bmi, d)

        return dict(
            coord=coord,  # (bm, dn)
            feature=feature,  # (bm, dim_out)
        )


class SPointCrossAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_qkv: int,
        cross_attn_type: str,
        num_heads: int = 8,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
    ):
        """
        Args:
            dim_q:
                dimension of the input query tokens
            dim_kv:
                dimension of the input kv tokens (tokens that will be attended to)
            dim_qkv:
                feature dimension of qkv.
                Since we want to use flash attention, force them to be the same.
            cross_attn_type:
                'global': typical global attention
                'localized_knn': assign kv to the cloest query
                'localized_voxel':  assign kv to the query in the same voxel
            num_heads:
                number of heads in the multihead attention.
            dropout_prob:
                dropout probability.
            use_rmsnorm:
                whether to use rms norm.
            add_bias:
                whether to add bias in linear qkv and output
        """
        super().__init__()
        assert dim_qkv % num_heads == 0
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_qkv = dim_qkv
        self.cross_attn_type = cross_attn_type
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = self.dim_qkv // self.num_heads
        self.dropout_prob = dropout_prob
        self.add_bias = add_bias

        # linear projection
        self.linear_q = _ops().Linear(
            in_features=self.dim_q,
            out_features=self.dim_qkv,
            bias=self.add_bias,
        )
        self.linear_kv = _ops().Linear(
            in_features=self.dim_kv,
            out_features=2 * self.dim_qkv,
            bias=self.add_bias,
        )
        self.linear_out = _ops().Linear(
            in_features=self.dim_qkv,
            out_features=self.dim_q,
            bias=self.add_bias,
        )
        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(self.dim_qkv)
            self.rmsnorm_k = RMSNorm(self.dim_qkv)

        # pre layer normalization
        self.layernorm_q = _ops().LayerNorm(self.dim_q)
        self.layernorm_kv = _ops().LayerNorm(self.dim_kv)

    def _forward_xformers(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        coord_query: ppoint.PackedPoint,
        coord_key: ppoint.PackedPoint,
        cell_width: T.Optional[float] = None,
        use_cached: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query:
                (bn, dim_q)
            key_value:
                (bm, dim_kv)
            coord_query:
                (bn, dn) packed
            coord_key:
                (bm, dn) packed
            cell_width:
                cell width used in voxel windowed cross attention

        Returns:
            (bn, dim_q)
        """

        bn, dim_q = query.shape
        bm, dim_kv = key_value.shape

        # pre layer norm
        query = self.layernorm_q(query)  # (bn, dq)
        key_value = self.layernorm_kv(key_value)  # (bm, dkv)

        # linear projection
        query = self.linear_q(query)  # (bn, dim_qkv)
        key_value = self.linear_kv(key_value)  # (bm, 2 * dim_qkv)
        key, value = torch.chunk(key_value, chunks=2, dim=-1)  # (bm, dim_qkv)

        if self.use_rmsnorm:
            query = self.rmsnorm_q(query)
            key = self.rmsnorm_k(key)

        # attention
        query = query.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dim_head)
        key = key.reshape(bm, self.num_heads, self.dim_head)  # (bm, h, dim_head)
        value = value.reshape(bm, self.num_heads, self.dim_head)  # (bm, h, dim_head)

        if self.cross_attn_type == "global":
            # each query can attend to all kv
            out = ppoint.cross_softmax_attention_with_packed_qkv(
                packed_query_coord=coord_query,  # (bn, dn)
                packed_query=query,  # (bn, h, dim_head)
                packed_kv_coord=coord_key,  # (bm, dn)
                packed_key=key,  # (bm, h, dim_head)
                packed_value=value,  # (bm, h, dim_head)
                save_to_cache=use_cached,
            )  # (bn, h, dim_qkv)
        elif self.cross_attn_type == "localized_knn":
            # each query attends to its nonoverlapping cluster
            out = ppoint.localized_knn_cross_softmax_attention_with_packed_qkv(
                packed_query_coord=coord_query,  # (bn, dn)
                packed_query=query,  # (bn, h, dim_head)
                packed_kv_coord=coord_key,  # (bm, dn)
                packed_key=key,  # (bm, h, dim_head)
                packed_value=value,  # (bm, h, dim_head)
                use_cached=use_cached,
                cache_name="localized_knn_cross",
                debug=debug,
            )  # (bn, h, dim_qkv)
        elif self.cross_attn_type == "localized_voxel":
            # each query attends to kv within the same voxel
            #
            # It is the responsibility of the one who creating the queries to
            # ensure queries cover the entire scene (all kvs).
            #
            # Note: we can have multiple queries in the same voxel (they simply attend to the same kv)
            assert cell_width is not None
            out = ppoint.voxel_windowed_cross_softmax_attention_with_packed_qkv(
                packed_query_coord=coord_query,  # (bn, dn)
                packed_query=query,  # (bn, h, dim_head)
                packed_kv_coord=coord_key,  # (bm, dn)
                packed_key=key,  # (bm, h, dim_head)
                packed_value=value,  # (bm, h, dim_head)
                cell_width=cell_width,
                use_cached=use_cached,
                cache_name="voxel_windowed_cross",
                debug=debug,
            )  # (bn, h, dim_qkv)
        else:
            raise NotImplementedError

        out = self.linear_out(out.reshape(bn, self.dim_qkv))  # (bn, dim_v)
        return out

    def _forward_flash(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        coord_query: ppoint.PackedPoint,
        coord_key: ppoint.PackedPoint,
        cell_width: T.Optional[float] = None,
        use_cached: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Use flash_attn_varlen_kvpacked_func as backend to compute cross attention.

        Args:
            query:
                (bn, dim_q)
            key_value:
                (bm, dim_kv)
            coord_query:
                (bn, dn) packed
            coord_key:
                (bm, dn) packed
            cell_width:
                cell width used in voxel windowed cross attention

        Returns:
            (bn, dim_q)
        """

        bn, dim_q = query.shape
        bm, dim_kv = key_value.shape

        # pre layer norm
        query = self.layernorm_q(query)  # (bn, dq)
        key_value = self.layernorm_kv(key_value)  # (bm, dkv)

        # linear projection
        query = self.linear_q(query)  # (bn, dim_qkv)
        key_value = self.linear_kv(key_value)  # (bm, 2 * dim_qkv)
        key_value = key_value.reshape(bm, 2, self.num_heads * self.dim_head)  # (bm, 2kv, dim_qkv)

        # if self.use_rmsnorm:
        #     query = self.rmsnorm_q(query)
        #     key_value = torch.stack(
        #         [
        #             self.rmsnorm_k(key_value[:, 0]),  # (bn, dim_qkv)
        #             key_value[:, 1],   # (bn, dim_qkv)
        #         ], dim=1)  # (bn, 2kv, dim_qkv)

        if self.use_rmsnorm:
            query = self.rmsnorm_q(query)  # (bn, dim_qkv)

            packed_k, packed_v = torch.chunk(key_value, chunks=2, dim=1)  # (bn, dim_qkv)
            packed_k = self.rmsnorm_k(packed_k)  # (bn, dim_qkv)

            query = query.reshape(bn, self.num_heads, self.dim_head)
            packed_k = packed_k.reshape(bm, self.num_heads, self.dim_head)
            packed_v = packed_v.reshape(bm, self.num_heads, self.dim_head)

            key_value = None
        else:
            query = query.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dim_head)
            key_value = key_value.reshape(bm, 2, self.num_heads, self.dim_head)  # (bm, 2kv, h, dim_head)
            packed_k = None
            packed_v = None

        # attention
        if self.cross_attn_type == "global":
            # each query can attend to all kv
            if key_value is not None:
                out = ppoint.cross_softmax_attention_with_packed_qkv_flash_stacked(
                    packed_query_coord=coord_query,  # (bn, dn)
                    packed_query=query,  # (bn, h, dim_head)
                    packed_kv_coord=coord_key,  # (bm, dn)
                    packed_kv=key_value,  # (bm, 2kv, h, dim_head)
                    save_to_cache=use_cached,
                )  # (bn, h, dim_qkv)
            else:
                out = ppoint.cross_softmax_attention_with_packed_qkv_flash(
                    packed_query_coord=coord_query,  # (bn, dn)
                    packed_query=query,  # (bn, h, dim_head)
                    packed_kv_coord=coord_key,  # (bm, dn)
                    packed_key=packed_k,  # (bm, h, dim_head)
                    packed_value=packed_v,  # (bm, h, dim_head)
                    save_to_cache=use_cached,
                )  # (bn, h, dim_qkv)

        elif self.cross_attn_type == "localized_knn":
            # each query attends to its nonoverlapping cluster
            if key_value is not None:
                out = ppoint.localized_knn_cross_softmax_attention_with_packed_qkv_flash_stacked(
                    packed_query_coord=coord_query,  # (bn, dn)
                    packed_query=query,  # (bn, h, dim_head)
                    packed_kv_coord=coord_key,  # (bm, dn)
                    packed_kv=key_value,  # (bm, 2kv, h, dim_head)
                    use_cached=use_cached,
                    cache_name="localized_knn_cross",
                    debug=debug,
                )  # (bn, h, dim_qkv)
            else:
                out = ppoint.localized_knn_cross_softmax_attention_with_packed_qkv_flash(
                    packed_query_coord=coord_query,  # (bn, dn)
                    packed_query=query,  # (bn, h, dim_head)
                    packed_kv_coord=coord_key,  # (bm, dn)
                    packed_key=packed_k,  # (bm, h, dim_head)
                    packed_value=packed_v,  # (bm, h, dim_head)
                    use_cached=use_cached,
                    cache_name="localized_knn_cross",
                    debug=debug,
                )  # (bn, h, dim_qkv)

        elif self.cross_attn_type == "localized_voxel":
            # each query attends to kv within the same voxel
            #
            # It is the responsibility of the one who creating the queries to
            # ensure queries cover the entire scene (all kvs).
            #
            # Note: we can have multiple queries in the same voxel (they simply attend to the same kv)
            assert cell_width is not None
            if key_value is not None:
                out = ppoint.voxel_windowed_cross_softmax_attention_with_packed_qkv_flash_stacked(
                    packed_query_coord=coord_query,  # (bn, dn)
                    packed_query=query,  # (bn, h, dim_head)
                    packed_kv_coord=coord_key,  # (bm, dn)
                    packed_kv=key_value,  # (bm, 2kv, h, dim_head)
                    cell_width=cell_width,
                    use_cached=use_cached,
                    cache_name="voxel_windowed_cross",
                    debug=debug,
                )  # (bn, h, dim_qkv)
            else:
                out = ppoint.voxel_windowed_cross_softmax_attention_with_packed_qkv_flash(
                    packed_query_coord=coord_query,  # (bn, dn)
                    packed_query=query,  # (bn, h, dim_head)
                    packed_kv_coord=coord_key,  # (bm, dn)
                    packed_key=packed_k,  # (bm, h, dim_head)
                    packed_value=packed_v,  # (bm, h, dim_head)
                    cell_width=cell_width,
                    use_cached=use_cached,
                    cache_name="voxel_windowed_cross",
                    debug=debug,
                )  # (bn, h, dim_qkv)

        else:
            raise NotImplementedError

        out = self.linear_out(out.reshape(bn, self.dim_qkv))  # (bn, dim_v)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        coord_query: ppoint.PackedPoint,
        coord_key: ppoint.PackedPoint,
        cell_width: T.Optional[float] = None,
        use_cached: bool = False,
        use_grad_checkpointing: bool = False,
        debug: bool = False,
        attn_backend: str = "flash",
    ) -> torch.Tensor:
        """
        Args:
            query:
                (bn, dim_q)
            key_value:
                (bm, dim_kv)
            coord_query:
                (bn, dn) packed
            coord_key:
                (bm, dn) packed
            cell_width:
                cell width used by voxel windowed cross attention
            attn_backend:
                "xformers"
                "flash"

        Returns:
            (bn, dim_q)
        """
        if use_grad_checkpointing:
            if attn_backend == "xformers":
                return torch.utils.checkpoint.checkpoint(
                    self._forward_xformers,
                    query,
                    key_value,
                    coord_query,
                    coord_key,
                    cell_width,
                    use_cached,
                    debug,
                    use_reentrant=False,
                )
            elif attn_backend == "flash":
                return torch.utils.checkpoint.checkpoint(
                    self._forward_flash,
                    query,
                    key_value,
                    coord_query,
                    coord_key,
                    cell_width,
                    use_cached,
                    debug,
                    use_reentrant=False,
                )
            else:
                raise NotImplementedError(attn_backend)
        else:
            if attn_backend == "xformers":
                return self._forward_xformers(
                    query=query,
                    key_value=key_value,
                    coord_query=coord_query,
                    coord_key=coord_key,
                    cell_width=cell_width,
                    use_cached=use_cached,
                    debug=debug,
                )
            elif attn_backend == "flash":
                return self._forward_flash(
                    query=query,
                    key_value=key_value,
                    coord_query=coord_query,
                    coord_key=coord_key,
                    cell_width=cell_width,
                    use_cached=use_cached,
                    debug=debug,
                )
            else:
                raise NotImplementedError(attn_backend)


class SPointSelfAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_qkv: int,
        self_attn_type: str,
        num_heads: int = 8,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
    ):
        """
        Args:
            dim_in:
                feature dimension of the input tokens
            dim_qkv:
                feature dimension of qkv.
                Since we want to use flash attention, force them to be the same.
            self_attn_type:
                'global': typical global attention
                'localized_knn': tokens within the same cluster attend to each other
                'localized_voxel': tokens within the same voxel attend to each other
            num_heads:
                number of heads in the multihead attention.
            dropout_prob:
                dropout probability.
            use_rmsnorm:
                whether to use rms norm.
            add_bias:
                whether to add bias in linear qkv and output
        """
        super().__init__()
        assert dim_qkv % num_heads == 0, f"{dim_qkv}, {num_heads}"
        self.dim_in = dim_in
        self.dim_qkv = dim_qkv
        self.self_attn_type = self_attn_type
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = self.dim_qkv // self.num_heads
        self.dropout_prob = dropout_prob
        self.add_bias = add_bias

        # linear projection
        self.linear_qkv = _ops().Linear(
            in_features=self.dim_in,
            out_features=3 * self.dim_qkv,
            bias=self.add_bias,
        )
        self.linear_out = _ops().Linear(
            in_features=self.dim_qkv,
            out_features=self.dim_in,
            bias=self.add_bias,
        )
        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(self.dim_qkv)
            self.rmsnorm_k = RMSNorm(self.dim_qkv)

    def _forward_xformers(
        self,
        x: torch.Tensor,  # (bn, d)
        coord: ppoint.PackedPoint,  # (bn, dn)
        k: T.Optional[T.Union[int, T.List[int]]] = None,  # (b,)
        cell_width: T.Optional[float] = None,
        shift_ratio: T.Union[float, torch.Tensor] = 0,
        debug: bool = False,
    ):
        bn, d = x.shape

        # project to qkv
        qkv = self.linear_qkv(x)  # (bn, 3*dim_qkv)
        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)  # (bn, dim_qkv)

        if self.use_rmsnorm:
            query = self.rmsnorm_q(query)  # (bn, dim_qkv)
            key = self.rmsnorm_k(key)  # (bn, dim_qkv)

        query = query.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dim_head)
        key = key.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dim_head)
        value = value.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dim_head)

        # attention
        if self.self_attn_type == "global":
            # each token attends to all other tokens
            # we reuse cross attention (setting qkv to be the same)
            out = ppoint.cross_softmax_attention_with_packed_qkv(
                packed_query_coord=coord,  # (bn, dn)
                packed_query=query,
                packed_kv_coord=coord,  # (bn, dn)
                packed_key=key,
                packed_value=value,
                save_to_cache=True,
            )  # (bn, h, dhead)
        elif self.self_attn_type == "localized_knn":
            # randomly divide tokens into k non-overlapping clusters.
            #
            # Note: k can be different for different samples in the batch
            assert key is not None
            out = ppoint.localized_knn_self_softmax_attention_packed(
                packed_coord=coord,  # (bn, dn)
                packed_query=query,
                packed_key=key,
                packed_value=value,
                k=k,
                use_cached=False,  # different cluster for different layer
                cache_name=None,
                printout=debug,
            )  # (bn, h, dhead)
        elif self.self_attn_type == "localized_voxel":
            # tokens in the same voxel attend to each other
            out = ppoint.voxel_windowed_self_softmax_attention(
                packed_coord=coord,
                packed_query=query,
                packed_key=key,
                packed_value=value,
                cell_width=cell_width,
                shift=shift_ratio * cell_width,
                save_to_cache=True,
                printout=debug,
            )  # (bn, h, dhead)
        else:
            raise NotImplementedError

        out = self.linear_out(out.reshape(bn, self.dim_qkv))  # (bn, dim_in)

        return out  # (bn, dim_in)

    def _forward_flash(
        self,
        x: torch.Tensor,  # (bn, d)
        coord: ppoint.PackedPoint,  # (bn, dn)
        k: T.Optional[T.Union[int, T.List[int]]] = None,  # (b,)
        cell_width: T.Optional[float] = None,
        shift_ratio: T.Union[float, torch.Tensor] = 0,
        debug: bool = False,
    ):
        bn, d = x.shape

        # project to qkv
        qkv = self.linear_qkv(x)  # (bn, 3qkv * dim_qkv)
        qkv = qkv.reshape(bn, 3, self.num_heads * self.dim_head)  # (bn, 3qkv, dim_qkv)

        if self.use_rmsnorm:
            packed_q, packed_k, packed_v = torch.chunk(qkv, chunks=3, dim=1)  # (bn, 1, dim_qkv)
            packed_q = self.rmsnorm_q(packed_q)  # (bn, 1, dim_qkv)
            packed_k = self.rmsnorm_k(packed_k)  # (bn, 1, dim_qkv)

            packed_q = packed_q.reshape(bn, self.num_heads, self.dim_head)  # (bn, num_head, dim_head)
            packed_k = packed_k.reshape(bn, self.num_heads, self.dim_head)  # (bn, num_head, dim_head)
            packed_v = packed_v.reshape(bn, self.num_heads, self.dim_head)  # (bn, num_head, dim_head)

            qkv = None
        else:
            qkv = qkv.reshape(bn, 3, self.num_heads, self.dim_head)  # (bn, 3qkv, num_head, dim_head)
            packed_q = None
            packed_k = None
            packed_v = None

        # attention
        if self.self_attn_type == "global":
            # each token attends to all other tokens
            if qkv is not None:
                out = ppoint.self_softmax_attention_with_packed_qkv_flash_stacked(
                    packed_coord=coord,  # (bn, dn)
                    packed_qkv=qkv,  # (bn, 3qkv, num_head, dim_head)
                )  # (bn, h, dhead)
            else:
                out = ppoint.self_softmax_attention_with_packed_qkv_flash(
                    packed_coord=coord,  # (bn, dn)
                    packed_query=packed_q,
                    packed_key=packed_k,
                    packed_value=packed_v,  # (bn, num_head, dim_head)
                )  # (bn, h, dhead)
        elif self.self_attn_type == "localized_knn":
            # randomly divide tokens into k non-overlapping clusters.
            #
            # Note: k can be different for different samples in the batch
            assert k is not None
            if qkv is not None:
                out = ppoint.localized_knn_self_softmax_attention_packed_flash_stacked(
                    packed_coord=coord,  # (bn, dn)
                    packed_qkv=qkv,  # (bn, 3qkv, num_head, dim_head)
                    k=k,
                    use_cached=False,  # different cluster for different layer
                    cache_name=None,
                    printout=debug,
                )  # (bn, h, dhead)
            else:
                out = ppoint.localized_knn_self_softmax_attention_packed_flash(
                    packed_coord=coord,  # (bn, dn)
                    packed_query=packed_q,
                    packed_key=packed_k,
                    packed_value=packed_v,
                    k=k,
                    use_cached=False,  # different cluster for different layer
                    cache_name=None,
                    printout=debug,
                )  # (bn, h, dhead)
        elif self.self_attn_type == "localized_voxel":
            # tokens in the same voxel attend to each other
            if qkv is not None:
                out = ppoint.voxel_windowed_self_softmax_attention_flash_stacked(
                    packed_coord=coord,  # (bn, dn)
                    packed_qkv=qkv,  # (bn, 3qkv, num_head, dim_head)
                    cell_width=cell_width,
                    shift=shift_ratio * cell_width,
                    save_to_cache=True,
                    printout=debug,
                )  # (bn, h, dhead)
            else:
                out = ppoint.voxel_windowed_self_softmax_attention_flash(
                    packed_coord=coord,  # (bn, dn)
                    packed_query=packed_q,
                    packed_key=packed_k,
                    packed_value=packed_v,
                    cell_width=cell_width,
                    shift=shift_ratio * cell_width,
                    save_to_cache=True,
                    printout=debug,
                )  # (bn, h, dhead)
        else:
            raise NotImplementedError

        out = self.linear_out(out.reshape(bn, self.dim_qkv))  # (bn, dim_in)

        return out  # (bn, dim_in)

    def forward(
        self,
        x: torch.Tensor,
        coord: ppoint.PackedPoint,
        k: T.Optional[T.Union[int, T.List[int]]] = None,
        cell_width: T.Optional[float] = None,
        shift_ratio: T.Union[float, torch.Tensor] = 0,
        use_grad_checkpointing: bool = False,
        debug: bool = False,
        attn_backend: str = "flash",
    ) -> torch.Tensor:
        """
        Args:
            x:
                (bn, dim_in) packed feature
            coord:
                (bn, dn), packed coordinate
            k:
                int or list of (b,), number of clusters.  when k = 1, it is equivalent to global attention.
                only needed if using `localized_knn`
            cell_width:
                float, cell width used by voxel_windowed
            shift_ratio:
                used by voxel window attention. It will shift by shift_ratio * cell_width.
            attn_backend:
                "xformers"
                "flash"

        Returns:
            (bn, dim_in)
        """

        if use_grad_checkpointing:
            if attn_backend == "flash":
                return torch.utils.checkpoint.checkpoint(
                    self._forward_flash,
                    x,  # (bn, d)
                    coord,  # (bn, dn)
                    k,
                    cell_width,
                    shift_ratio,
                    debug,
                    use_reentrant=False,
                )
            elif attn_backend == "xformers":
                return torch.utils.checkpoint.checkpoint(
                    self._forward_xformers,
                    x,  # (bn, d)
                    coord,  # (bn, dn)
                    k,
                    cell_width,
                    shift_ratio,
                    debug,
                    use_reentrant=False,
                )
            else:
                raise NotImplementedError
        else:
            if attn_backend == "flash":
                return self._forward_flash(
                    x=x,  # (bn, d)
                    coord=coord,  # (bn, dn)
                    k=k,
                    cell_width=cell_width,
                    shift_ratio=shift_ratio,
                    debug=debug,
                )
            elif attn_backend == "xformers":
                return self._forward_xformers(
                    x=x,  # (bn, d)
                    coord=coord,  # (bn, dn)
                    k=k,
                    cell_width=cell_width,
                    shift_ratio=shift_ratio,
                    debug=debug,
                )
            else:
                raise NotImplementedError


class SPointPerceiverEncoderBlock(torch.nn.Module):
    """
    Each perceiver encoder block is composed of
    1. cross attention (q: latent (packed point) -> kv: tokens (packed point))
        'global': typical global attention
        'localized_knn': assign kv to the cloest query
        'localized_voxel':  assign kv to the query in the same voxel
    2. self attention (q: latent (packed point) -> kv: latent (packed point))
        'global': typical global attention
        'localized_knn': tokens within the same cluster attend to each other
        'localized_voxel': tokens within the same voxel attend to each other
    3. mlp
    """

    def __init__(
        self,
        dim_latent: int,  # query
        dim_token: int,  # kv
        dim_qkv: int,
        cross_attn_type: str,  # see above comments for options
        self_attn_type: str,  # see above comments for options
        num_self_attn: int = 2,
        num_self_heads: int = 8,
        num_cross_heads: int = 8,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 4,
        mlp_type: str = "swiglu",
        linear_in_attn_add_bias: bool = False,
        mlp_add_bias: bool = False,
        add_kv_linear: bool = False,  # because we use prenorm (a layernorm before linear layers in multihead attetion), we might consider adding linear layers
    ):
        super().__init__()
        self.cross_attn_type = cross_attn_type
        self.self_attn_type = self_attn_type
        self.add_kv_linear = add_kv_linear

        if self.add_kv_linear:
            self.kv_linear = _ops().Linear(
                in_features=dim_token,
                out_features=dim_token,
                bias=False,  # followed by layernorm
            )
        else:
            self.kv_linear = None

        self.ca_ln = _ops().LayerNorm(dim_latent, eps=1e-6)

        self.ca_layer = SPointCrossAttentionLayer(
            dim_q=dim_latent,
            dim_kv=dim_token,
            dim_qkv=dim_qkv,
            cross_attn_type=self.cross_attn_type,
            num_heads=num_cross_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=use_rmsnorm,
            add_bias=linear_in_attn_add_bias,
        )

        mlp_hidden_dim = int(dim_latent * mlp_ratio)
        if mlp_type == "timm":
            approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
            self.ca_mlp = Mlp(
                in_features=dim_latent,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                bias=mlp_add_bias,
            )
        elif mlp_type == "swiglu":
            self.ca_mlp = _SwiGLU(
                in_features=dim_latent,
                hidden_features=mlp_hidden_dim,
                out_features=None,
                bias=mlp_add_bias,
            )
        else:
            raise NotImplementedError

        # self attention blocks
        _self_attention_layers = []
        _mlp_layers = []
        _ln1_layers = []
        _ln2_layers = []
        for _ in range(num_self_attn):
            ln1 = _ops().LayerNorm(dim_latent, eps=1e-6)
            ln2 = _ops().LayerNorm(dim_latent, eps=1e-6)
            _ln1_layers.append(ln1)
            _ln2_layers.append(ln2)

            sa_layer = SPointSelfAttentionLayer(
                dim_in=dim_latent,
                dim_qkv=dim_qkv,
                self_attn_type=self.self_attn_type,
                num_heads=num_self_heads,
                dropout_prob=dropout_prob,
                use_rmsnorm=use_rmsnorm,
                add_bias=linear_in_attn_add_bias,
            )
            _self_attention_layers.append(sa_layer)

            mlp_hidden_dim = int(dim_latent * mlp_ratio)
            if mlp_type == "timm":
                approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
                mlp_layer = Mlp(
                    in_features=dim_latent,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                )
            elif mlp_type == "swiglu":
                mlp_layer = _SwiGLU(
                    in_features=dim_latent,
                    hidden_features=mlp_hidden_dim,
                    out_features=None,
                    bias=mlp_add_bias,
                )
            else:
                raise NotImplementedError
            _mlp_layers.append(mlp_layer)

        self.ln1_layers = torch.nn.ModuleList(_ln1_layers)
        self.ln2_layers = torch.nn.ModuleList(_ln2_layers)
        self.sa_layers = torch.nn.ModuleList(_self_attention_layers)
        self.mlp_layers = torch.nn.ModuleList(_mlp_layers)

    def forward(
        self,
        query: torch.Tensor,  # packed
        key_value: torch.Tensor,  # packed
        coord_query: ppoint.PackedPoint,
        coord_key: ppoint.PackedPoint,
        k: T.Optional[T.Union[int, T.List[int]]],  # for knn self attention
        cross_cell_width: T.Optional[float],
        self_cell_width: T.Optional[float],
        use_grad_checkpointing: bool = False,
        debug: bool = False,
    ):
        r"""
        Args:
            query:
                (bn, dim_latent) packed feature
            key_value:
                (bm, dim_kv)  packed feature
            coord_query:
                (bn, dn)
            coord_key:
                (bm, dn)
            k:
                int or list of (b,), number of clusters / patches.  1: typical global attention.
                Increasing k increases locality
            cross_cell_width:
                cell width used in voxel windowed cross attention
            self_cell_width:
                cell width used in voxel windowed self attention

        Returns:
            latents:
                (bn, dim_latent)
        """

        if self.kv_linear is not None:
            key_value = self.kv_linear(key_value)  # (bm, dim_kv)

        # cross attention (latent -> input token)
        query = query + self.ca_layer(
            query=query,
            key_value=key_value,
            coord_query=coord_query,
            coord_key=coord_key,
            cell_width=cross_cell_width,
            use_cached=True,
            use_grad_checkpointing=use_grad_checkpointing,
            debug=debug,
        )  # (b, n, dim_latent)
        query = query + self.ca_mlp(self.ca_ln(query))  # (b, n, dim_latent)

        # self attention (latent -> latent)
        for i, (ln1, sa_layer, ln2, mlp_layer) in enumerate(
            zip(
                self.ln1_layers,
                self.sa_layers,
                self.ln2_layers,
                self.mlp_layers,
            )
        ):
            query = query + sa_layer(
                x=ln1(query),
                coord=coord_query,
                k=k,
                cell_width=self_cell_width,
                shift_ratio=0.5 * (i % 2),
                use_grad_checkpointing=use_grad_checkpointing,
                debug=debug,
            )  # (b, n, dim_latent)
            query = query + mlp_layer(ln2(query))  # (b, n, dim_latent)

        return query  # (b, n, dim_latent)


class SPointPerceiverEncoder(torch.nn.Module):
    """
    Perceiver encoder that takes a set of input tokens as kv and
    a set of latent tokens as query. For each layer, the latent tokens use
    cross attention to gather info from input tokens, then
    perform self attention among latents.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        num_blocks: int,
        dim_qkv: int,
        cross_attn_type: str,
        self_attn_type: str,
        num_in_cluster: T.Union[int, T.List[int]] = None,
        num_clusters: T.Union[int, T.List[int]] = None,
        cross_cell_widths: T.Union[float, T.List[float]] = None,
        self_cell_widths: T.Union[float, T.List[float]] = None,
        num_self_attn: int = 2,
        num_self_heads: int = 8,
        num_cross_heads: int = 8,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 4,
        mlp_type: str = "swiglu",
        linear_in_attn_add_bias: bool = False,
        mlp_add_bias: bool = False,
        add_kv_linear: bool = False,
    ):
        """
        Args:
            dim_latent:
                feature dimension of the latent vectors
            dim_token:
                feature dimension of the input tokens
            num_blocks:
                number of encoder blocks to use
            dim_qkv:
                dimension of the qkv used in cross and self attention
            cross_attn_type:
                'global': typical global attention
                'localized_knn': assign kv to the cloest query
                'localized_voxel:  assign kv to the query in the same voxel (no two queries in the same voxel)
            self_attn_type:
                'global': typical global attention
                'localized_knn': tokens within the same cluster attend to each other
                'localized_voxel': tokens within the same voxel attend to each other
            num_in_cluster:
                int or (num_blocks,) number of clusters = total points / num_in_cluster
            num_clusters:
                int or (num_blocks,), if given, ignore k_ratio and directly set it as the number of clusters
            cross_cell_widths:
                float or (num_blocks,), cell width used by voxel windowed cross attention
            self_cell_widths:
                float or (num_blocks,), cell width used by voxel windowed self attention
            num_self_attn:
                number of self attention in each encoder block
            num_self_heads:
                number of self attention heads in each encoder block
            num_cross_heads
                number of cross attention heads in each encoder block
            dropout_prob:
                dropout prob
            use_rmsnorm:
                whether to use rmsnorm (normalize the mean and std during cross and
                self attention at the output of the Wq, Wk, Wv)
            mlp_ratio:
                mlp expansion ratio
            add_kv_linear:
                whether to add a linear layer to input tokens (kv)

        Notes:
            1. when using voxel windowed cross attention, it is important that
            the queries are constructed by larger cells than the input points.
        """
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_token = dim_token
        self.num_blocks = num_blocks
        self.num_in_cluster = num_in_cluster
        if self.num_in_cluster is not None:
            if isinstance(self.num_in_cluster, int):
                self.num_in_cluster = [self.num_in_cluster] * num_blocks
        self.num_clusters = num_clusters
        if self.num_clusters is not None and isinstance(self.num_clusters, int):
            self.num_clusters = [self.num_clusters] * num_blocks

        self.cross_attn_type = cross_attn_type
        self.self_attn_type = self_attn_type
        self.cross_cell_widths = cross_cell_widths
        if self.cross_cell_widths is not None:
            if isinstance(self.cross_cell_widths, (int, float)):
                self.cross_cell_widths = [self.cross_cell_widths] * num_blocks
        self.self_cell_widths = self_cell_widths
        if self.self_cell_widths is not None:
            if isinstance(self.self_cell_widths, (int, float)):
                self.self_cell_widths = [self.self_cell_widths] * num_blocks

        # encoder blocks
        self.blocks = []
        for _layer_idx in range(num_blocks):
            self.blocks.append(
                SPointPerceiverEncoderBlock(
                    dim_latent=dim_latent,
                    dim_token=self.dim_token,
                    dim_qkv=dim_qkv,
                    cross_attn_type=self.cross_attn_type,
                    self_attn_type=self.self_attn_type,
                    num_self_attn=num_self_attn,
                    num_self_heads=num_self_heads,
                    num_cross_heads=num_cross_heads,
                    dropout_prob=dropout_prob,
                    use_rmsnorm=use_rmsnorm,
                    mlp_ratio=mlp_ratio,
                    mlp_type=mlp_type,
                    linear_in_attn_add_bias=linear_in_attn_add_bias,
                    mlp_add_bias=mlp_add_bias,
                    add_kv_linear=add_kv_linear,
                )
            )
        self.blocks = torch.nn.ModuleList(self.blocks)

    def set_num_clusters(self, num_clusters: T.Union[int, T.List[int]]):
        if num_clusters is not None and isinstance(num_clusters, int):
            num_clusters = [num_clusters] * self.num_blocks
        self.num_clusters = num_clusters

    def convert_num_in_cluster_to_num_clusters(self, ref_num_points: int):
        """Converts num_in_cluster (which determines number of points in a cluster)
        to number of clusters."""
        # num_cluster = ref_num_points * k_ratio
        self.num_clusters = [max(1, round(ref_num_points / num_in_cluster)) for num_in_cluster in self.num_in_cluster]

    def forward(
        self,
        input_tokens: torch.Tensor,  # kv
        latent_tokens: torch.Tensor,  # query
        coord_input_tokens: ppoint.PackedPoint,  # only need when using localized attention
        coord_latents: ppoint.PackedPoint,  # needed if using localized cross or self attention
        use_grad_checkpointing: bool = False,
        debug: bool = False,
    ) -> T.Dict[str, T.List[torch.Tensor]]:
        r"""
        Args:
            input_tokens:
                (bm, dim_token), key/value
            latent_tokens:
                (bn, dim_latent), query
            coord_input_tokens:
                (bm, dn)
            coord_latents:
                (bn, dn)

        Returns:
            latent_tokens:
                (bn, dim_latent) final layer's latent output
        """

        for i, block in enumerate(self.blocks):
            if self.num_clusters is not None:
                _num_cluster = self.num_clusters[i]
            elif self.num_in_cluster is not None:
                _num_cluster = [
                    max(1, round(seq_len / self.num_in_cluster[i])) for seq_len in coord_latents.seq_lens.tolist()
                ]
            else:
                _num_cluster = None

            latent_tokens = block(
                query=latent_tokens,  # (bn, d), query
                key_value=input_tokens,  # (bm, d), kv
                coord_query=coord_latents,  # (bn, dn)
                coord_key=coord_input_tokens,  # (bm, dn)
                k=_num_cluster,  # for self attn
                cross_cell_width=self.cross_cell_widths[i] if self.cross_cell_widths is not None else None,
                self_cell_width=self.self_cell_widths[i] if self.self_cell_widths is not None else None,
                use_grad_checkpointing=use_grad_checkpointing,
                debug=debug,
            )
            if debug:
                assert torch.isfinite(latent_tokens).all(), (
                    f"nan: {torch.isnan(latent_tokens).any()} inf: {torch.isinf(latent_tokens).any()}"
                )
        return latent_tokens


class SPointEncoder(torch.nn.Module):
    r"""
    Architecture overview:

    4M points -> MLP -> downsample -> 400k points -> select k points (voxel or random)
    -> cross attention (voxel or knn) -> self attention (voxel or knn or global)
    -> optionally cross attention to convert to global
    """

    def __init__(
        self,
        coord_inputs: T.Union[T.List[str], str],  # 'xyz', 'rgb', 'tao'
        # mlp (include downsampling)
        mlp_inputs: T.Union[T.List[str], str],
        mlp_config_dict: T.Dict[str, T.Any],  # target, params
        # init perceiver query
        init_query_method: str,  # see below for options
        init_query_config: T.Dict[str, T.Any],
        # perceiver
        perceiver_inputs: T.Union[T.List[str], str],
        perceiver_config_dict: T.Dict[str, T.Any],  # target, params
        # vperceiver (to convert set latent to vector latent)
        vperceiver_init_query_method: str,  # see below for options
        vperceiver_init_query_config: T.Dict[str, T.Any],
        vperceiver_inputs: T.Union[T.List[str], str],
        vperceiver_config_dict: T.Dict[str, T.Any],
        # final layer
        dim_output: int,
        #
        convert_output_to_batch_format: bool,
        # positional encoding
        # legacy: it is cumbersome to manually add arguments if additional positional encodings are needed.
        xyz_pos_encoding_config: T.Dict[str, T.Any] = None,  # target, params, ignored if pos_encoding_configs is given
        rgb_pos_encoding_config: T.Dict[str, T.Any] = None,  # ignored if pos_encoding_configs is given
        normal_pos_encoding_config: T.Dict[str, T.Any] = None,  # ignored if pos_encoding_configs is given
        tao_pos_encoding_config: T.Dict[str, T.Any] = None,  # target, params, ignored if pos_encoding_configs is given
        # new: make pos_encoding configuations in one place
        pos_encoding_configs: T.Dict[str, T.Dict[str, T.Any]] | None = None,
        #
        use_fp32_for_final_layer: bool = False,
    ):
        """
        Args:
            coord_inputs:
                what to include in the point coord.
                'xyz', 'rgb', 'tao',
            mlp_inputs:
                what to include as the input to mlp.
                'xyz', 'rgb', 'normal', 'tao', 'feature'
                'encoded_xyz', 'encoded_rgb', 'encoded_normal', 'encoded_tao'
            dim_output:
                dimension of the final output

            init_query_method:
                'learned'
                'subsample'
                'subsample_avg_coord_avg_feature'
                'subsample_avg_coord_learned_feature'
                "voxel_subsample"
                "voxel_avg_coord_avg_feature"
                "voxel_avg_coord_learned_feature"

            init_query_config:
                num_latent:
                    needed by 'learned', 'subsample', 'subsample_avg_coord_avg_feature',
                    'random_avg_coord_learned_feature'
                cell_width:
                    needed by "voxel_subsample", "voxel_avg_coord_avg_feature",
                    "voxel_avg_coord_learned_feature"
                fps_multiplier:
                    if not None or -1, use farthest point sampling.
                    First we subsample `num_latent * fps_multiplier` of points from input points
                    and then use fps to sample the `num_latent` points.
                    Note: Using fps during training makes it slow!
                fps_min_num_points:
                    min number of points to first select when doing fps.
                    -1: no minimum

            perceiver_inputs:
                what to include as the input to perceiver
                'xyz', 'rgb', 'normal', 'tao', 'feature' (output of mlp if existed, else input of mlp)
                'encoded_xyz', 'encoded_rgb', 'encoded_normal', 'encoded_tao'

            vperceiver_init_query_method:
                same options as init_query_method
            vperceiver_init_query_config:
                same options as init_query_config
            vperceiver_inputs:
                what to include as the input to vperceiver
                'xyz', 'rgb', 'normal', 'tao', 'feature' (perceiver output)
                'encoded_xyz', 'encoded_rgb', 'encoded_normal', 'encoded_tao'

            convert_output_to_batch_format:
                whether to convert output to batch (b, n, d) format.
                only possible if `init_query_method` is the following:
                'learned'
                'subsample'
                'subsample_avg_coord_avg_feature'
                'subsample_avg_coord_learned_feature'
                of if vperceiver is used and the above is used as the vperceiver init method.

            use_fp32_for_final_layer:
                if true, we make sure the linear layer is performed in full precision.
        """

        super().__init__()
        if isinstance(coord_inputs, str):
            coord_inputs = [coord_inputs]
        self.coord_inputs = coord_inputs
        assert "xyz" in self.coord_inputs

        if isinstance(mlp_inputs, str):
            mlp_inputs = [mlp_inputs]
        self.mlp_inputs = mlp_inputs

        if isinstance(perceiver_inputs, str):
            perceiver_inputs = [perceiver_inputs]
        self.perceiver_inputs = perceiver_inputs

        self.dim_output = dim_output

        # construct positional encoding
        if pos_encoding_configs is None:
            # legacy
            self.pos_encoder_dict = None

            if xyz_pos_encoding_config is not None:
                self.xyz_pos_encoder = config_utils.instantiate_from_config(xyz_pos_encoding_config)
            else:
                self.xyz_pos_encoder = None
            self.xyz_pos_encoder_is_used = False

            if rgb_pos_encoding_config is not None:
                self.rgb_pos_encoder = config_utils.instantiate_from_config(rgb_pos_encoding_config)
            else:
                self.rgb_pos_encoder = None
            self.rgb_pos_encoder_is_used = False

            if normal_pos_encoding_config is not None:
                self.normal_pos_encoder = config_utils.instantiate_from_config(normal_pos_encoding_config)
            else:
                self.normal_pos_encoder = None
            self.normal_pos_encoder_is_used = False

            if tao_pos_encoding_config is not None:
                self.tao_pos_encoder = config_utils.instantiate_from_config(tao_pos_encoding_config)
            else:
                self.tao_pos_encoder = None
            self.tao_pos_encoder_is_used = False
        else:
            self.pos_encoder_dict = torch.nn.ModuleDict()
            self.pos_encoder_used_dict = dict()
            for tmp_k, tmp_config in pos_encoding_configs.items():
                self.pos_encoder_dict[tmp_k] = config_utils.instantiate_from_config(tmp_config)
                self.pos_encoder_used_dict[tmp_k] = False

        self.dim_coord = self._compute_input_dim(
            input_names=self.coord_inputs,
        )

        # mlp
        self.dim_mlp_input = self._compute_input_dim(
            input_names=self.mlp_inputs,
        )
        self.mlp_config_dict = mlp_config_dict
        if not self.mlp_config_dict is None:
            if self.mlp_config_dict["target"].endswith("PointwiseResnetVoxelBlocks"):
                # add input dimensino
                self.mlp_config_dict["params"]["dim_in"] = self.dim_mlp_input
                self.mlp_config_dict["params"]["dim_coord"] = self.dim_coord
                self.mlp = config_utils.instantiate_from_config(self.mlp_config_dict)
                self.mlp_dim_out = self.mlp.dim_out
            else:
                raise NotImplementedError(self.mlp_config_dict)
        else:
            self.mlp = None
            self.mlp_dim_out = self.dim_mlp_input

        # get perceiver kv dim
        for name in self.perceiver_inputs:
            assert name in self.coord_inputs + ["feature"] + [f"encoded_{n}" for n in self.coord_inputs], (
                f"{self.perceiver_inputs}"
            )
        self.dim_perceiver_kv = self._compute_input_dim(
            input_names=self.perceiver_inputs,
            dim_feature=self.mlp_dim_out,
        )

        # get perceiver main
        self.perceiver_config_dict = perceiver_config_dict
        if self.perceiver_config_dict["target"].endswith("SPointPerceiverEncoder"):
            self.perceiver_dim = self.perceiver_config_dict["params"]["dim_latent"]  # main dimension of the perceiver
            self.perceiver_config_dict["params"]["dim_token"] = self.dim_perceiver_kv
            self.perceiver = config_utils.instantiate_from_config(self.perceiver_config_dict)
        else:
            raise NotImplementedError(self.perceiver_config_dict)

        # perceiver init query
        # 'learned', 'subsample', 'subsample_avg_coord_avg_feature', 'random_avg_coord_learned_feature', "voxel_subsample", "voxel_avg_coord_avg_feature", "voxel_avg_coord_learned_feature",
        self.init_query_method = init_query_method
        self.init_query_config = init_query_config
        self.perceiver_init_query_linear = None
        self.latent_constructor = None
        if self.init_query_method == "learned":
            assert self.init_query_config.get("num_latent", None) is not None
            self.latent_constructor = ShapeLatent(
                num_latent=self.init_query_config["num_latent"],
                dim_latent=self.perceiver_dim,
            )
        elif self.init_query_method in ["random_avg_coord_learned_feature", "voxel_avg_coord_learned_feature"]:
            self.latent_constructor = ShapeLatent(
                num_latent=1,
                dim_latent=self.perceiver_dim,
            )
        elif self.init_query_method in [
            "subsample",
            "subsample_avg_coord_avg_feature",
            "voxel_subsample",
            "voxel_avg_coord_avg_feature",
        ]:
            if self.dim_perceiver_kv != self.perceiver_dim:
                self.perceiver_init_query_linear = _ops().Linear(
                    in_features=self.dim_perceiver_kv,
                    out_features=self.perceiver_dim,
                    bias=True,
                )
        else:
            raise NotImplementedError

        self.convert_output_to_batch_format = convert_output_to_batch_format
        if self.convert_output_to_batch_format:
            assert self.init_query_method in [
                "learned",
                "subsample",
                "subsample_avg_coord_avg_feature",
                "subsample_avg_coord_learned_feature",
            ], f"{self.init_query_method} is not supported converting to batch format"

        # vperceiver (convert set perceiver output to vector output)
        self.vperceiver_config_dict = vperceiver_config_dict
        self.vperceiver_init_query_method = vperceiver_init_query_method
        self.vperceiver_init_query_config = vperceiver_init_query_config
        self.vperceiver_inputs = vperceiver_inputs
        self.vperceiver_init_query_linear = None
        self.vperceiver_latent_constructor = None
        if self.vperceiver_config_dict is not None:
            # get vperceiver kv dim
            for name in self.vperceiver_inputs:
                assert name in self.coord_inputs + ["feature"] + [f"encoded_{n}" for n in self.coord_inputs], (
                    f"{self.vperceiver_inputs}"
                )
            self.dim_vperceiver_kv = self._compute_input_dim(
                input_names=self.vperceiver_inputs,
                dim_feature=self.perceiver_dim,
            )

            # construct the vperceiver network
            if self.vperceiver_config_dict["target"].endswith("SPointPerceiverEncoder"):
                self.vperceiver_dim = self.perceiver_config_dict["params"][
                    "dim_latent"
                ]  # main dimension of the perceiver
                self.vperceiver_config_dict["params"]["dim_token"] = self.dim_vperceiver_kv
                self.vperceiver = config_utils.instantiate_from_config(self.vperceiver_config_dict)
            else:
                raise NotImplementedError(self.perceiver_config_dict)

            # init query for vperceiver
            if self.vperceiver_init_query_method == "learned":
                assert self.vperceiver_init_query_config.get("num_latent", None) is not None
                self.vperceiver_latent_constructor = ShapeLatent(
                    num_latent=self.vperceiver_init_query_config["num_latent"],
                    dim_latent=self.vperceiver_dim,
                )
            elif self.vperceiver_init_query_method in [
                "random_avg_coord_learned_feature",
                "voxel_avg_coord_learned_feature",
            ]:
                self.vperceiver_latent_constructor = ShapeLatent(
                    num_latent=1,
                    dim_latent=self.vperceiver_dim,
                )
            elif self.vperceiver_init_query_method in [
                "subsample",
                "subsample_avg_coord_avg_feature",
                "voxel_subsample",
                "voxel_avg_coord_avg_feature",
            ]:
                if self.dim_perceiver_kv != self.vperceiver_dim:
                    self.vperceiver_init_query_linear = _ops().Linear(
                        in_features=self.dim_perceiver_kv,
                        out_features=self.vperceiver_dim,
                        bias=True,
                    )
            else:
                raise NotImplementedError

        else:
            self.dim_vperceiver_kv = None
            self.vperceiver = None
            self.vperceiver_dim = self.perceiver_dim

        # output layer
        self.use_fp32_for_final_layer = use_fp32_for_final_layer
        self.final_layer = _ops().Linear(
            in_features=self.vperceiver_dim,
            out_features=self.dim_output,
        )

        # remove unused positional encoding
        if pos_encoding_configs is None:
            if not self.xyz_pos_encoder_is_used:
                self.xyz_pos_encoder = None
            if not self.rgb_pos_encoder_is_used:
                self.rgb_pos_encoder = None
            if not self.normal_pos_encoder_is_used:
                self.normal_pos_encoder = None
            if not self.tao_pos_encoder_is_used:
                self.tao_pos_encoder = None
        else:
            for tmp_k in list(self.pos_encoder_dict.keys()):
                if not self.pos_encoder_used_dict[tmp_k]:
                    del self.pos_encoder_dict[tmp_k]

    def _compute_input_dim_single_mode(
        self,
        *,
        mode: str,
        dim_point_token: int,
    ) -> int:
        """
        Compute the encoded dimension of an input signal (ie, after positional encoding).
        The positional encoding is stored in `self.pos_encoder_dict`.

        Returns:
            int, dimension of the encoded signal.
        """
        assert mode in self.pos_encoder_dict, f"{mode} {list(self.pos_encoder_dict.keys())=}"
        dim_point_token += self.pos_encoder_dict[mode].dim_out
        self.pos_encoder_used_dict[mode] = True
        return dim_point_token

    def _compute_input_dim(
        self,
        input_names: T.List[str],
        dim_feature: int = None,
    ):
        # compute input dimension
        dim_point_token = 0
        for key in input_names:
            if key == "xyz":
                dim_point_token += 3
            elif key == "rgb":
                dim_point_token += 3
            elif key == "albedo":
                dim_point_token += 3
            elif key == "alpha":
                dim_point_token += 1
            elif key == "roughness_metallic":
                dim_point_token += 2
            elif key == "normal":
                dim_point_token += 3
            elif key == "tao":
                dim_point_token += 1
            elif key == "plucker":
                dim_point_token += 6
            elif key == "ray_o":
                dim_point_token += 3
            elif key == "ray_d":
                dim_point_token += 3
            elif key == "ray_origin_direction_w":
                dim_point_token += 6
            elif key == "encoded_xyz":
                if self.pos_encoder_dict is None:
                    # legacy
                    assert self.xyz_pos_encoder is not None
                    dim_point_token += self.xyz_pos_encoder.dim_out
                    self.xyz_pos_encoder_is_used = True
                else:
                    dim_point_token = self._compute_input_dim_single_mode(mode="xyz", dim_point_token=dim_point_token)
            elif key == "encoded_rgb":
                if self.pos_encoder_dict is None:
                    # legacy
                    assert self.rgb_pos_encoder is not None
                    dim_point_token += self.rgb_pos_encoder.dim_out
                    self.rgb_pos_encoder_is_used = True
                else:
                    dim_point_token = self._compute_input_dim_single_mode(mode="rgb", dim_point_token=dim_point_token)
            elif key == "encoded_normal":
                if self.pos_encoder_dict is None:
                    # legacy
                    assert self.xyz_pos_encoder is not None
                    dim_point_token += self.xyz_pos_encoder.dim_out
                    self.xyz_pos_encoder_is_used = True
                else:
                    dim_point_token = self._compute_input_dim_single_mode(
                        mode="normal", dim_point_token=dim_point_token
                    )
            elif key == "encoded_tao":
                if self.pos_encoder_dict is None:
                    # legacy
                    assert self.tao_pos_encoder is not None
                    dim_point_token += self.tao_pos_encoder.dim_out
                    self.tao_pos_encoder_is_used = True
                else:
                    dim_point_token = self._compute_input_dim_single_mode(mode="tao", dim_point_token=dim_point_token)
            elif key in [
                "encoded_plucker",
                "encoded_ray_o",
                "encoded_ray_d",
                "encoded_ray_origin_direction_w",
                "encoded_albedo",
                "encoded_alpha",
                "encoded_roughness_metallic",
            ]:
                assert self.pos_encoder_dict is not None
                dim_point_token = self._compute_input_dim_single_mode(
                    mode=key.split("encoded_")[1],
                    dim_point_token=dim_point_token,
                )
            elif key == "feature":
                assert dim_feature is not None
                dim_point_token += dim_feature
            else:
                raise NotImplementedError(f"{key=}")
        return dim_point_token

    def _construct_input(
        self,
        input_names: T.List[str],
        xyz_w: T.Optional[torch.Tensor] = None,  # (b, n, 3xyz_w) or (bn, 3xyz_w)
        rgb: T.Optional[torch.Tensor] = None,  # (b, n, 3rgb) or (bn, 3rgb) [-1, 1]
        normal_w: T.Optional[torch.Tensor] = None,  # (b, n, 3xyz_w) or (bn, 3xyz_w)
        tao: T.Optional[torch.Tensor] = None,  # (b, n, 1) or (bn, 1)
        ray_origin_direction_w: T.Optional[torch.Tensor] = None,  # (b, n, 6xyz_w) origin->direction
        plucker: T.Optional[torch.Tensor] = None,  # (b, n, 6) or (bn, 6)
        albedo: T.Optional[torch.Tensor] = None,  # (b, n, 3rgb) or (bn, 3rgb) [-1, 1]
        roughness_metallic: T.Optional[torch.Tensor] = None,  # (b, n, 2) or (bn, 2) [-1, 1]
        alpha: T.Optional[torch.Tensor] = None,  # (b, n, 1) or (bn, 1) [-1, 1]
        feature: T.Optional[torch.Tensor] = None,  # (b, n, dim_point_feature)  or (bn, d)
        pcd_other_attrs: T.Dict[str, torch.Tensor] = {},
    ):
        """
        construct the input feature or input coord.
        return (b, n, d) or (bn, d)
        """
        if input_names is None or len(input_names) == 0:
            return None, dict()

        input_token = []
        dim_start_dict = dict()
        dim_end_dict = dict()
        current_dim = 0
        for key in input_names:
            if key == "xyz":
                assert xyz_w is not None
                input_token.append(xyz_w)
                dim_start_dict[key] = current_dim
                current_dim += 3
                dim_end_dict[key] = current_dim
            elif key == "rgb":
                assert rgb is not None
                input_token.append(rgb)
                dim_start_dict[key] = current_dim
                current_dim += 3
                dim_end_dict[key] = current_dim
            elif key == "normal":
                assert normal_w is not None
                input_token.append(normal_w)
                dim_start_dict[key] = current_dim
                current_dim += 3
                dim_end_dict[key] = current_dim
            elif key == "tao":
                assert tao is not None
                input_token.append(tao)
                dim_start_dict[key] = current_dim
                current_dim += 1
                dim_end_dict[key] = current_dim
            elif key == "plucker":
                if plucker is not None:
                    arr = plucker
                else:
                    assert key in pcd_other_attrs, f"{key=}, {list(pcd_other_attrs.keys())=}"
                    arr = pcd_other_attrs[key]
                assert arr.size(-1) == 6
                input_token.append(arr)
                dim_start_dict[key] = current_dim
                current_dim += 6
                dim_end_dict[key] = current_dim
            elif key == "ray_o":
                assert key in pcd_other_attrs, f"{key=}, {list(pcd_other_attrs.keys())=}"
                arr = pcd_other_attrs[key]
                assert arr.size(-1) == 3
                input_token.append(arr)
                dim_start_dict[key] = current_dim
                current_dim += 3
                dim_end_dict[key] = current_dim
            elif key == "ray_d":
                assert key in pcd_other_attrs, f"{key=}, {list(pcd_other_attrs.keys())=}"
                arr = pcd_other_attrs[key]
                assert arr.size(-1) == 3
                input_token.append(arr)
                dim_start_dict[key] = current_dim
                current_dim += 3
                dim_end_dict[key] = current_dim
            elif key == "ray_origin_direction_w":
                assert ray_origin_direction_w is not None
                input_token.append(ray_origin_direction_w)
                dim_start_dict[key] = current_dim
                current_dim += 6
                dim_end_dict[key] = current_dim
            elif key == "albedo":
                assert albedo is not None
                input_token.append(albedo)
                dim_start_dict[key] = current_dim
                current_dim += 3
                dim_end_dict[key] = current_dim
            elif key == "alpha":
                assert alpha is not None
                input_token.append(alpha)
                dim_start_dict[key] = current_dim
                current_dim += 1
                dim_end_dict[key] = current_dim
            elif key == "roughness_metallic":
                assert roughness_metallic is not None
                input_token.append(roughness_metallic)
                dim_start_dict[key] = current_dim
                current_dim += 2
                dim_end_dict[key] = current_dim
            elif key == "encoded_xyz":
                assert xyz_w is not None
                if self.pos_encoder_dict is None:
                    # legacy
                    input_token.append(self.xyz_pos_encoder(xyz_w))
                else:
                    input_token.append(self.pos_encoder_dict["xyz"](xyz_w))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_rgb":
                assert rgb is not None
                if self.pos_encoder_dict is None:
                    input_token.append(self.rgb_pos_encoder(rgb))
                else:
                    input_token.append(self.pos_encoder_dict["rgb"](rgb))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_albedo":
                assert albedo is not None
                input_token.append(self.pos_encoder_dict["albedo"](albedo))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_alpha":
                assert alpha is not None
                input_token.append(self.pos_encoder_dict["alpha"](alpha))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_roughness_metallic":
                assert roughness_metallic is not None
                input_token.append(self.pos_encoder_dict["roughness_metallic"](roughness_metallic))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_normal":
                assert normal_w is not None
                if self.pos_encoder_dict is None:
                    raise NotImplementedError
                else:
                    if isinstance(self.pos_encoder_dict["normal"], PluckerEmbed):
                        input_token.append(
                            self.pos_encoder_dict["normal"](
                                torch.cat(
                                    [
                                        xyz_w,  # (bn, 3)
                                        normal_w,  # (bn, 3)
                                    ],
                                    dim=-1,
                                )
                            )
                        )  # (bn, 6)
                    else:
                        input_token.append(self.pos_encoder_dict["normal"](normal_w))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_tao":
                assert tao is not None
                if self.pos_encoder_dict is None:
                    input_token.append(self.tao_pos_encoder(tao))
                else:
                    input_token.append(self.pos_encoder_dict["tao"](tao))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "encoded_ray_origin_direction_w":
                assert ray_origin_direction_w is not None
                input_token.append(self.pos_encoder_dict["ray_origin_direction_w"](ray_origin_direction_w))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key in ["encoded_plucker", "encoded_ray_o", "encoded_ray_d"]:
                key_raw = key.split("encoded_")[1]
                assert key_raw in pcd_other_attrs, f"{key_raw=}, {list(pcd_other_attrs.keys())=}"
                input_token.append(self.pos_encoder_dict[key_raw](pcd_other_attrs[key_raw]))
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            elif key == "feature":
                assert feature is not None
                input_token.append(feature)
                dim_start_dict[key] = current_dim
                current_dim += input_token[-1].size(-1)
                dim_end_dict[key] = current_dim
            else:
                raise NotImplementedError
        input_token = (
            torch.cat(input_token, dim=-1) if len(input_token) > 1 else input_token[0]
        )  # (b, n, dim_point_token)
        return input_token, dim_start_dict, dim_end_dict

    def _construct_init_query(
        self,
        init_query_method: str,
        init_query_config: T.Dict[str, T.Any],
        coord: ppoint.PackedPoint,
        feature: torch.Tensor,
        latent_constructor: T.Optional = None,
        point_probs: T.Optional[torch.Tensor] = None,
        num_latent: T.Optional[int] = None,
    ):
        """
        Construct the initial query of the perceiver from the output of the MLP.

        Args:
            coord:
                (n1+n2+...+nb, dn)
            coord_dim_start_dict:
                where each signal lies in coord
            feature:
                (n1+n2+...+nb, d)
            point_probs:
                (n1+n2+...+nb,) or None. When using "subsample" query method, determines the probability of a point being used as a query. If None, defaults to equal probability.
            num_latent:
                int or None. Number of latents to use. If None, uses the one provided in the initial query config.

        Returns:
            latent_coord:
                (m1+m2+...+mb, dn) or None if learned
            latent:
                (m1+m2+...+mb, d)
            out_type:
                'learned'
                'subsampled'
                'voxel'
        """

        b = coord.batch_size
        dn = coord.dn
        d = feature.size(-1)

        # get perceiver initial query
        if init_query_method == "learned":
            assert latent_constructor is not None
            assert init_query_config.get("num_latent", None) is not None

            latent = latent_constructor(batch_size=b)  # (b, num_latent, dim_perceiver)
            latent_seq_lens = [latent.size(1)] * b
            latent = latent.reshape(b * latent.size(1), latent.size(2))  # (b * num_latent, dim_perceiver)
            latent_coord = ppoint.PackedPoint(
                coord=torch.zeros(latent.size(0), dn, dtype=coord.dtype, device=coord.device),  # dummy
                seq_lens=latent_seq_lens,
            )  # (b * num_latent, dn)
            out_type = "learned"

        elif init_query_method == "subsample":
            # randomly pick with either equal chance or chance provided by point_probs, use the selected coordinate and feature
            if num_latent is None:
                assert init_query_config.get("num_latent", None) is not None
                num_latent = init_query_config.get("num_latent")

            # since we do not expect the batch size to be large,
            # let's do the easiest thing (one sample at a time with for loop)
            # instead of overengineering things
            seq_lens = coord.seq_lens  # (b,)
            current_idx = 0
            latent = []
            latent_coord = []  # (b, num_latent, dn)
            latent_seq_lens = []
            for ib in range(b):
                seqlen = seq_lens[ib]

                if (
                    init_query_config.get("fps_multiplier", None) is not None
                    and init_query_config.get("fps_multiplier", -1) > 0
                    and point_probs is not None
                ):
                    # use fps
                    # num subset to work on
                    num_subset = max(
                        int(num_latent * init_query_config["fps_multiplier"]),
                        int(init_query_config.get("fps_min_num_points", -1)),
                    )
                    ridxs = torch.randperm(seqlen, device=coord.device)[:num_subset] + current_idx  # (num_subset,)
                    with _nullcontext():
                        _coord, _idx = pytorch3d.ops.sample_farthest_points(
                            points=coord.coord[ridxs].unsqueeze(0).float(),  # (1, num_subset, dn)
                            lengths=None,
                            K=num_latent,
                            random_start_point=False,  # _ridxs is already randomly selected
                        )
                    _coord = _coord.squeeze(0)  # (num_latent, dn)
                    _idx = _idx.squeeze(0)  # (num_latent,)
                    latent_seq_lens.append(_coord.size(0))
                    latent_coord.append(_coord)  # (num_latent, dn)
                    latent.append(feature[ridxs[_idx]])  # (num_latent, d)
                    del _coord
                    del _idx
                else:
                    if point_probs is None:
                        # random sample with uniform probabilities
                        ridxs = torch.randperm(seqlen, device=coord.device)[:num_latent] + current_idx
                    else:
                        # random sample according to probabilities
                        ridxs = (
                            torch.multinomial(
                                torch.clamp(point_probs[current_idx : current_idx + seqlen], min=1e-6),
                                num_samples=num_latent,
                                replacement=False,
                            )
                            + current_idx
                        )
                    latent_seq_lens.append(ridxs.size(0))
                    latent_coord.append(coord.coord[ridxs])  # (num_latent, dn)
                    latent.append(feature[ridxs])  # (num_latent, d)

                current_idx += seqlen
            latent = torch.cat(latent, dim=0) if len(latent) > 0 else latent[0]  # (b * num_latent, d)
            latent_coord = ppoint.PackedPoint(
                coord=torch.cat(latent_coord, dim=0)
                if len(latent_coord) > 0
                else latent_coord[0],  # (b * num_latent, dn)
                seq_lens=latent_seq_lens,
            )
            out_type = "subsampled"

        elif init_query_method in [
            "subsample_avg_coord_avg_feature",
            "subsample_avg_coord_learned_feature",
        ]:
            # randomly pick with equal chance, use the selected coordinate and feature
            if num_latent is None:
                assert init_query_config.get("num_latent", None) is not None
                num_latent = init_query_config.get("num_latent")

            # since we do not expect the batch size to be large,
            # let's do the easiest thing (one sample at a time with for loop)
            # instead of overengineering things
            seq_lens = coord.seq_lens  # (b,)
            current_idx = 0
            latent = []
            latent_coord = []
            latent_seq_lens = []
            for ib in range(b):
                seqlen = seq_lens[ib]

                if (
                    init_query_config.get("fps_multiplier", None) is not None
                    and init_query_config.get("fps_multiplier", -1) > 0
                ):
                    # use fps
                    # num subset to work on
                    num_subset = max(
                        int(num_latent * init_query_config["fps_multiplier"]),
                        int(init_query_config.get("fps_min_num_points", -1)),
                    )
                    ridxs = torch.randperm(seqlen, device=coord.device)[:num_subset] + current_idx  # (num_subset,)
                    with _nullcontext():
                        _coord, _idx = pytorch3d.ops.sample_farthest_points(
                            points=coord.coord[ridxs].unsqueeze(0).float(),  # (1, num_subset, dn)
                            lengths=None,
                            K=num_latent,
                            random_start_point=False,  # _ridxs is already randomly selected
                        )
                    _idx = _idx.squeeze(0)  # (num_latent,)
                    ridxs = ridxs[_idx]  # (num_latent,)
                    del _coord
                    del _idx
                else:
                    # random sample with uniform probabilities
                    ridxs = torch.randperm(seqlen, device=coord.device)[:num_latent] + current_idx  # (num_latent,)

                latent_seq_lens.append(ridxs.size(0))

                s_coord = coord.coord[current_idx : current_idx + seqlen]  # (n, dn)
                r_coord = coord.coord[ridxs]  # (num_latent, dn)
                with _nullcontext():
                    knn_out = pytorch3d.ops.knn_points(
                        p1=s_coord.unsqueeze(0).float(),  # (1, n, dn)
                        p2=r_coord.unsqueeze(0).float(),  # (1, m, dn) query
                        K=1,
                    )
                    _kidxs = knn_out.idx.squeeze(-1).squeeze(0)  # (n, )

                # compute averaged coordinate
                _latent_coord = torch.zeros_like(r_coord)  # (num_latent, dn)
                _latent_coord.scatter_reduce_(
                    dim=0,
                    index=_kidxs.unsqueeze(-1).expand(-1, s_coord.size(-1)),  # (n, dn)
                    src=s_coord,  # (n, dn)
                    reduce="mean",
                    include_self=False,
                )  # (num_latent, dn)
                latent_coord.append(_latent_coord)  # (num_latent, dn)

                if init_query_method == "subsample_avg_coord_avg_feature":
                    _latent_feature = torch.zeros(
                        _latent_coord.size(0), d, dtype=feature.dtype, device=feature.device
                    )  # (num_latent, d)
                    _latent_feature.scatter_reduce_(
                        dim=0,
                        index=_kidxs.unsqueeze(-1).expand(-1, d),  # (n, d)
                        src=feature[current_idx : current_idx + seqlen],  # (n, d)
                        reduce="mean",
                        include_self=False,
                    )  # (num_latent, d)
                    latent.append(_latent_feature)  # (num_latent, d)

                elif init_query_method == "random_avg_coord_learned_feature":
                    assert latent_constructor is not None
                    _latent_feature = latent_constructor(batch_size=1)  # (b=1, num_latent=1, d)
                    assert _latent_feature.shape == (1, 1, d), f"{_latent_feature.shape} != (1, 1, {d})"
                    _latent_feature = _latent_feature.reshape(1, d).expand(_latent_coord.size(0), d)  # (num_latent, d)
                    latent.append(_latent_feature)  # (num_latent, d)

                else:
                    raise NotImplementedError

                current_idx += seqlen

            latent = torch.cat(latent, dim=0) if len(latent) > 0 else latent[0]  # (b * num_latent, d)
            latent_coord = ppoint.PackedPoint(
                coord=torch.cat(latent_coord, dim=0)
                if len(latent_coord) > 0
                else latent_coord[0],  # (b * num_latent, dn)
                seq_lens=latent_seq_lens,
            )
            out_type = "subsampled"

        elif init_query_method in [
            "voxel_subsample",
            "voxel_avg_coord_avg_feature",
            "voxel_avg_coord_learned_feature",
        ]:
            # init query is constructed by voxelization and randomly select one in the voxel
            assert init_query_config.get("cell_width", None) is not None
            cell_width = init_query_config.get("cell_width")
            if init_query_method == "voxel_subsample":
                out_dict = ppoint.voxel_downsampling(
                    packed_coord=coord,
                    packed_feature=feature,
                    cell_width=cell_width,
                    shift=0,
                    save_to_cache=False,
                    aggregation_method="subsample",
                )
                latent_coord = out_dict["packed_coord"]  # (num_occupied_cells, dn)
                latent = out_dict["packed_feature"]  # (num_occupied_cells, d)

            elif init_query_method in [
                "voxel_avg_coord_avg_feature",
                "voxel_avg_coord_learned_feature",
            ]:
                out_dict = ppoint.voxel_downsampling(
                    packed_coord=coord,
                    packed_feature=feature,
                    cell_width=cell_width,
                    shift=0,
                    save_to_cache=False,
                    aggregation_method="mean",
                )
                latent_coord = out_dict["packed_coord"]  # (num_occupied_cells, dn)
                latent = out_dict["packed_feature"]  # (num_occupied_cells, d)

                if init_query_method == "voxel_avg_coord_learned_feature":
                    # replace with learned feature
                    assert latent_constructor is not None
                    latent = latent_constructor(batch_size=1)  # (b=1, num_latent=1, d)
                    assert latent.shape == (1, 1, d), f"{latent.shape} != (1, 1, {d})"
                    latent = latent.reshape(1, d).expand(latent_coord.bn, d)  # (num_latent, d)

            else:
                raise NotImplementedError

            out_type = "voxel"

        else:
            raise NotImplementedError

        return dict(
            latent_coord=latent_coord,  # (b * num_latent, dn)
            latent=latent,  # (b * num_latent, d)
            out_type=out_type,
        )

    def run_perceiver(
        self,
        perceiver: torch.nn.Module,
        source_coord: ppoint.PackedPoint,  # input coord  (bn', dn) used to construct query
        source_token: torch.Tensor,  # (bn', d)  used to construct query
        kv_coord: ppoint.PackedPoint,  # input coord  (bn, dn)   used as kv
        kv_token: torch.Tensor,  # (bn, d)  used as kv
        init_query_method: str,
        init_query_config: T.Dict[str, T.Any],
        latent_constructor: T.Optional[torch.nn.Module],
        init_query_linear: T.Optional[torch.nn.Module],
        use_grad_checkpointing: bool,
        debug: bool = False,
        point_probs: T.Optional[
            torch.Tensor
        ] = None,  # (bn,) probability of each point for importance sampling, None for equal
        num_latent: T.Optional[
            int
        ] = None,  # number of latents to use, None to default to the one provided by initial query config
    ):
        """
        Returns:
            out_tokens:
                (bl, d) latent tokens
            out_coord:
                (bl, dn) latent coord
            out_type:
                'learned', 'subsampled', 'voxel'
        """

        # construct init latent (query)
        _out_dict = self._construct_init_query(
            init_query_method=init_query_method,
            init_query_config=init_query_config,
            coord=source_coord,
            feature=source_token,
            latent_constructor=latent_constructor,
            point_probs=point_probs,
            num_latent=num_latent,
        )
        latent_coord: ppoint.PackedPoint = _out_dict["latent_coord"]  # (b * num_latent, dn)
        latent_tokens: torch.Tensor = _out_dict["latent"]  # (b * num_latent, d)
        out_type = _out_dict["out_type"]

        if init_query_linear is not None:
            latent_tokens = init_query_linear(latent_tokens)  # (b * num_latent, d)

        # run perceiver
        if isinstance(perceiver, SPointPerceiverEncoder):
            latent_tokens = perceiver(
                input_tokens=kv_token,  # kv, (bn, d)
                latent_tokens=latent_tokens,  # q,  (bl, d)
                coord_input_tokens=kv_coord,  # (bn, dn)
                coord_latents=latent_coord,  # (bl, dn)
                use_grad_checkpointing=use_grad_checkpointing,
                debug=debug,
            )  # (bl, d)
        else:
            raise NotImplementedError

        return dict(
            out_tokens=latent_tokens,  # (bl, d)
            out_coord=latent_coord,  # (bl, dn)
            out_type=out_type,
        )

    def forward(
        self,
        # point
        xyz_w: T.Optional[torch.Tensor] = None,  # (b, n, 3xyz_w)
        rgb: T.Optional[torch.Tensor] = None,  # (b, n, 3rgb)  [-1, 1]
        normal_w: T.Optional[torch.Tensor] = None,  # (b, n, 3xyz_w)
        ray_origin_direction_w: T.Optional[torch.Tensor] = None,  # (b, n, 6_ro_rd)
        albedo: T.Optional[torch.Tensor] = None,  # (b, n, 3rgb)  [-1, 1]
        roughness_metallic: T.Optional[torch.Tensor] = None,  # (b, n, 2)  [-1, 1]
        alpha: T.Optional[torch.Tensor] = None,  # (b, n, 1)  [-1, 1]
        pcd_other_attrs: T.Dict[str, torch.Tensor] = dict(),
        point_probs: T.Optional[
            torch.Tensor
        ] = None,  # (b, n) probability of each point for importance sampling, None for equal
        tao: T.Optional[torch.Tensor] = None,  # (b, n, 1)
        use_grad_checkpointing: bool = False,
        debug: bool = False,
        num_latent: T.Optional[int] = None,  # number of latents to use, None to default to initial query config
    ):
        """
        Args:
            xyz_w:
                (b, n, 3xyz_w) or None. point xyz
            rgb:
                (b, n, 3rgb) or None. [-1, 1]
            normal_w:
                (b, n, 3xyz_w) or None.
            pcd_other_attrs:
                (b, n, dim), other attributes for point cloud.
            tao:
                (b, n, 1) or None.  point timestamp
            point_probs:
                (b, n) or None. When using "subsample" query method, determines the probability of a point being used as a query. If None, defaults to equal probability.
            num_latent:
                int or None, number of latents to use when using subsample method. If None, defaults to the initialization provided in the config.

        Returns:
            latent_tokens:
                (b, num_latent, dim_output) or (bl, dim_output) depending on whether
                it is possible to convert from packed format back to batch format
            latent_coord:
                (b, num_latent, dn) tensor if batch format
                (bl, dn) packed point if packed format
                None, if learnable queries
            format:
                str:  'packed' or 'batch'
        """

        # construct coord
        coord, coord_dim_start_dict, coord_dim_end_dict = self._construct_input(
            input_names=self.coord_inputs,
            xyz_w=xyz_w,
            rgb=rgb,
            normal_w=normal_w,
            tao=tao,
            ray_origin_direction_w=ray_origin_direction_w,
            albedo=albedo,
            roughness_metallic=roughness_metallic,
            alpha=alpha,
            pcd_other_attrs=pcd_other_attrs,
        )  # (b, n, dn)
        b, n, dn = coord.shape

        # construct input for mlp
        input_token, _, _ = self._construct_input(
            input_names=self.mlp_inputs,
            xyz_w=xyz_w,
            rgb=rgb,
            normal_w=normal_w,
            tao=tao,
            ray_origin_direction_w=ray_origin_direction_w,
            albedo=albedo,
            roughness_metallic=roughness_metallic,
            alpha=alpha,
            pcd_other_attrs=pcd_other_attrs,
        )  # (b, n, dim_mlp_input)

        # convert to packed format
        coord = ppoint.PackedPoint(
            coord=coord.reshape(b * n, dn),
            seq_lens=[n] * b,
        )  # (bn, dn)
        input_token = input_token.reshape(b * n, input_token.size(-1))  # (bn, d)
        if point_probs is not None:
            point_probs = point_probs.reshape(b * n)

        # run mlp
        if self.mlp is not None:
            out_dict = self.mlp(
                coord=coord,  # (bn, dn) packed
                feature=input_token,  # (bn, d)
                use_grad_checkpointing=use_grad_checkpointing,
            )
            coord = out_dict["coord"]  # (bn', dn)
            input_token = out_dict["feature"]  # (bn', d)

        # construct perceiver kv tokens
        kv_token, _, _ = self._construct_input(
            input_names=self.perceiver_inputs,
            xyz_w=coord.coord[..., coord_dim_start_dict["xyz"] : coord_dim_end_dict["xyz"]]
            if coord_dim_start_dict.get("xyz", None) is not None
            else None,
            rgb=coord.coord[..., coord_dim_start_dict["rgb"] : coord_dim_end_dict["rgb"]]
            if coord_dim_start_dict.get("rgb", None) is not None
            else None,
            normal_w=coord.coord[..., coord_dim_start_dict["normal"] : coord_dim_end_dict["normal"]]
            if coord_dim_start_dict.get("normal", None) is not None
            else None,
            tao=coord.coord[..., coord_dim_start_dict["tao"] : coord_dim_end_dict["tao"]]
            if coord_dim_start_dict.get("tao", None) is not None
            else None,
            ray_origin_direction_w=coord.coord[
                ..., coord_dim_start_dict["ray_origin_direction_w"] : coord_dim_end_dict["ray_origin_direction_w"]
            ]
            if coord_dim_start_dict.get("ray_origin_direction_w", None) is not None
            else None,
            albedo=coord.coord[..., coord_dim_start_dict["albedo"] : coord_dim_end_dict["albedo"]]
            if coord_dim_start_dict.get("albedo", None) is not None
            else None,
            roughness_metallic=coord.coord[
                ..., coord_dim_start_dict["roughness_metallic"] : coord_dim_end_dict["roughness_metallic"]
            ]
            if coord_dim_start_dict.get("roughness_metallic", None) is not None
            else None,
            alpha=coord.coord[..., coord_dim_start_dict["alpha"] : coord_dim_end_dict["alpha"]]
            if coord_dim_start_dict.get("alpha", None) is not None
            else None,
            feature=input_token,
            pcd_other_attrs={
                k: coord.coord[..., coord_dim_start_dict[k] : coord_dim_end_dict[k]]
                for k in pcd_other_attrs
                if coord_dim_start_dict.get(k, None) is not None
            },
        )  # (bn', d)

        # run perceiver
        out_dict = self.run_perceiver(
            perceiver=self.perceiver,
            source_coord=coord,
            source_token=kv_token,
            kv_coord=coord,
            kv_token=kv_token,
            init_query_method=self.init_query_method,
            init_query_config=self.init_query_config,
            latent_constructor=self.latent_constructor,
            init_query_linear=self.perceiver_init_query_linear,
            use_grad_checkpointing=use_grad_checkpointing,
            debug=debug,
            point_probs=point_probs,
            num_latent=num_latent,
        )
        latent_tokens = out_dict["out_tokens"]  # (bl, d)
        latent_coord = out_dict["out_coord"]  # (bl, dn)
        out_type = out_dict["out_type"]

        # run vperceiver
        if self.vperceiver is not None:
            # construct vperceiver kv tokens (from current latent)
            vperceiver_kv_token, _ = self._construct_input(
                input_names=self.vperceiver_inputs,
                xyz_w=latent_coord.coord[..., coord_dim_start_dict["xyz"] : coord_dim_end_dict["xyz"]]
                if coord_dim_start_dict.get("xyz", None) is not None
                else None,
                rgb=latent_coord.coord[..., coord_dim_start_dict["rgb"] : coord_dim_end_dict["rgb"]]
                if coord_dim_start_dict.get("rgb", None) is not None
                else None,
                normal_w=latent_coord.coord[..., coord_dim_start_dict["normal"] : coord_dim_end_dict["normal"]]
                if coord_dim_start_dict.get("normal", None) is not None
                else None,
                tao=latent_coord.coord[..., coord_dim_start_dict["tao"] : coord_dim_end_dict["tao"]]
                if coord_dim_start_dict.get("tao", None) is not None
                else None,
                ray_origin_direction_w=latent_coord.coord[
                    ..., coord_dim_start_dict["ray_origin_direction_w"] : coord_dim_end_dict["ray_origin_direction_w"]
                ]
                if coord_dim_start_dict.get("ray_origin_direction_w", None) is not None
                else None,
                albedo=latent_coord.coord[..., coord_dim_start_dict["albedo"] : coord_dim_end_dict["albedo"]]
                if coord_dim_start_dict.get("albedo", None) is not None
                else None,
                roughness_metallic=latent_coord.coord[
                    ..., coord_dim_start_dict["roughness_metallic"] : coord_dim_end_dict["roughness_metallic"]
                ]
                if coord_dim_start_dict.get("roughness_metallic", None) is not None
                else None,
                alpha=latent_coord.coord[..., coord_dim_start_dict["alpha"] : coord_dim_end_dict["alpha"]]
                if coord_dim_start_dict.get("alpha", None) is not None
                else None,
                feature=latent_tokens,
                pcd_other_attrs={
                    k: latent_coord.coord[..., coord_dim_start_dict[k] : coord_dim_end_dict[k]]
                    for k in pcd_other_attrs
                    if coord_dim_start_dict.get(k, None) is not None
                },
            )  # (bn', d)

            # query of vperceiver is constructed from input points (input kv of perceiver)
            out_dict = self.run_perceiver(
                perceiver=self.vperceiver,
                source_coord=coord,
                source_token=kv_token,
                kv_coord=latent_coord,  # (bl, dn)
                kv_token=vperceiver_kv_token,  # (bl, d)
                init_query_method=self.vperceiver_init_query_method,
                init_query_config=self.vperceiver_init_query_config,
                latent_constructor=self.vperceiver_latent_constructor,
                init_query_linear=self.vperceiver_init_query_linear,
                use_grad_checkpointing=use_grad_checkpointing,
                debug=debug,
                point_probs=point_probs,
            )
            vperceiver_latent_tokens = out_dict["out_tokens"]  # (bl, d)
            vperceiver_latent_coord = out_dict["out_coord"]  # (bl, dn)
            out_type = out_dict["out_type"]
        else:
            vperceiver_latent_tokens = latent_tokens
            vperceiver_latent_coord = latent_coord

        # convert out type
        if self.convert_output_to_batch_format:
            if out_type == "learned":
                num_latent = vperceiver_latent_coord.bn // vperceiver_latent_coord.batch_size
                vperceiver_latent_tokens = vperceiver_latent_tokens.reshape(
                    vperceiver_latent_coord.batch_size, num_latent, vperceiver_latent_tokens.size(-1)
                )
                vperceiver_latent_coord = None
                format = "batch"
            elif out_type == "subsampled":
                num_latent = vperceiver_latent_coord.bn // vperceiver_latent_coord.batch_size
                vperceiver_latent_tokens = vperceiver_latent_tokens.reshape(
                    vperceiver_latent_coord.batch_size, num_latent, vperceiver_latent_tokens.size(-1)
                )
                vperceiver_latent_coord = vperceiver_latent_coord.coord.reshape(
                    vperceiver_latent_coord.batch_size, num_latent, vperceiver_latent_coord.dn
                )
                format = "batch"
            elif out_type == "voxel":
                raise RuntimeError
            else:
                raise NotImplementedError
        else:
            format = "packed"

        # final layer
        if self.final_layer is not None:
            if self.use_fp32_for_final_layer:
                with _nullcontext():
                    vperceiver_latent_tokens = self.final_layer(
                        vperceiver_latent_tokens.float()
                    )  # (b, num_latent, dim_latent) or (bl, dim_latent)
            else:
                vperceiver_latent_tokens = self.final_layer(
                    vperceiver_latent_tokens
                )  # (b, num_latent, dim_latent) or (bl, dim_latent)

        return dict(
            latent_coord=vperceiver_latent_coord,  # (bl, dn) or (b, num_latent, dn) or  None
            latent_tokens=vperceiver_latent_tokens,  # (bl, dim_latent) or (b, num_latent, dim_latent)
            format=format,
        )

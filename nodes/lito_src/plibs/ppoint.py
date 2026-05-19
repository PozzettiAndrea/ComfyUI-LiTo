#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements packed point.


import math
import typing as T

from timeit import default_timer as timer

import flash_attn

# Legacy alias kept so the dead "xformers" code paths below still parse; xops is
# never called now that ATTN_BACKEND defaults to "flash". The xformers->flash
# rewrite re-routes BlockDiagonal attention through plibs.flash_utils +
# flash_attn.flash_attn_varlen_func.
xops = None

import pytorch3d.ops
import torch

from plibs.linalg_utils import repeat_interleave
from plibs.flash_utils import create_block_diagonal_attn_bias_from_seq_lens
from contextlib import nullcontext as _nullcontext

if torch.cuda.is_available():
    import comfy.model_management as _comfy_mm
    device = _comfy_mm.get_torch_device()
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        print(f"GPU Name: {prop.name}")
        print(f"Compute Capability: {prop.major}.{prop.minor}")

# flash_attn supports Ampere+ (sm_80) which covers our entire target hardware
# matrix. The legacy "xformers" backend in this file is no longer reachable.
ATTN_BACKEND = "flash"


class PackedPoint:
    """
    Packed tensor for a batch of point clouds containing different numbers of points.

    - coord:
        (n1 + n2 + ... + nb, dn), dn can be 3d, 4d, nd
    - seq_lens:
        (b,) number of points in each batch
    - cache:
        a dict containing useful information that can be reused
    - device:
        PyTorch device.
    """

    def __init__(
        self,
        coord: torch.Tensor,  # (n1 + n2 + ... + nb, dn)
        seq_lens: T.Union[T.List[int], torch.Tensor],  # (b,)
        cache: T.Dict[str, T.Any] = None,
        coord_lim: torch.Tensor = None,  # (dn, 2), 2 for min/max for each coordinate
        device: torch.device = None,
    ):
        device = device if device is not None else coord.device
        if not isinstance(seq_lens, torch.Tensor):
            seq_lens = torch.tensor(seq_lens, dtype=torch.long, device=device)

        if coord is not None:
            self._coord = coord.to(device)  # (n1 + n2 + ... + nb, dn)
        else:
            self._coord = torch.zeros(torch.sum(seq_lens), device=device)
        self._seq_lens = seq_lens  # (b,)
        self.cache = cache
        self._coord_lim = coord_lim  # (dn, 2)  min/max

        assert self.bn == torch.sum(self._seq_lens), f"{self._coord.shape=}, {torch.sum(seq_lens)=}"

    def to(self, device: torch.device):
        self._coord = self._coord.to(device)
        self._seq_lens = self._seq_lens.to(device)
        if self._coord_lim is not None:
            self._coord_lim = self._coord_lim.to(device)
        if self.cache is not None:
            for k in self.cache:
                if isinstance(self.cache[k], torch.Tensor):
                    self.cache[k] = self.cache[k].to(device)

    def get_cache(self, name: str):
        if self.cache is not None and name in self.cache:
            return self.cache[name]
        return None

    def insert_cache(self, name: str, content: T.Any):
        if self.cache is None:
            self.cache = dict()
        if self.cache is not None:
            self.cache[name] = content

    def __repr__(self):
        return f"PackedPoint({self._coord}, {self._seq_lens})"

    def __str__(self):
        return self.__repr__()

    @property
    def dtype(self):
        return self._coord.dtype

    @property
    def device(self):
        return self._coord.device

    @property
    def coord(self):
        return self._coord  # (n1 + n2 + ... + nb, dn)

    @coord.setter
    def coord(self, coord: torch.Tensor):
        """Setter method for the value property"""
        # You can add validation logic here
        self._coord = coord
        self.cache = None  # need to reset cache since changing coordinates invalids the cache

    @property
    def coord_lim(self):
        return self._coord_lim  # (dn, 2min/max)

    @coord_lim.setter
    def coord_lim(self, coord_lim: torch.Tensor):
        """Setter method for the value property.

        Args:
            coord_lim:
                (dn, 2) min, max
        """
        self._coord_lim = coord_lim
        self.cache = None  # need to reset cache since changing coordinates invalids the cache

    @property
    def batch_size(self):
        return len(self.seq_lens)

    @property
    def bn(self):
        return self.coord.size(0)

    @property
    def dn(self):
        return self.coord.size(-1)

    @property
    def seq_lens(self):
        return self._seq_lens  # (b,)

    @property
    def batch_start_idxs(self):
        seq_lens = self.seq_lens  # (b,)
        batch_start_idxs = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens.dtype, device=seq_lens.device),
                seq_lens.cumsum(dim=0),  # (b,)
            ],
            dim=0,
        )  # (b+1,)
        return batch_start_idxs  # (b+1,)

    @property
    def batch_idxs(self):
        """
        Get batch indexes (n1 + n2 + ... + nb,)
        """
        bidx = torch.arange(self.batch_size, device=self.device)  # (b,)
        seq_lens = self.seq_lens  # (b,)

        # Repeat each index i by count[i] times
        batch_idxs = repeat_interleave(bidx, seq_lens, dim=0, output_size=self.bn)  # (bn,)
        return batch_idxs

    def index_batches(
        self,
        bs: T.Union[T.List[int], torch.Tensor],
        x: T.Optional[torch.Tensor] = None,
        unflatten: bool = False,
    ):
        """
        Selects batches based on their indices.

        Args:
            bs:
                (i, ) List of batch indices.
            x:
                (n1 + n2 + ... + nb, d) A packed tensor that matches the packing of the current PackedPoint. If not None, this function indexes this packed tensor rather than the original PackedPoint.

        Returns:
            (ni + nj + ..., d) Selected batches.

        """
        start_idx = self.batch_start_idxs[bs]
        end_idx = start_idx + self.seq_lens[bs]
        if unflatten:
            if x is not None:
                return [x[sidx:eidx] for sidx, eidx in zip(start_idx, end_idx)]
            else:
                return [self.coord[sidx:eidx] for sidx, eidx in zip(start_idx, end_idx)]
        else:
            if x is not None:
                return index_ranges(x, zip(start_idx, end_idx), dim=0)  # , self.seq_lens[bs]
            else:
                ncoord = index_ranges(self.coord, zip(start_idx, end_idx), dim=0)
                return PackedPoint(coord=ncoord, seq_lens=self.seq_lens[bs], cache=self.cache, coord_lim=self.coord_lim)

    def index_mask(
        self,
        mask: torch.Tensor,
        x: T.Optional[torch.Tensor] = None,
    ):
        cx = x if x is not None else self.coord
        output_x = []
        output_seq_lens = []
        for ib in range(self.batch_size):
            start_idx = self.batch_start_idxs[ib]
            end_idx = start_idx + self.seq_lens[ib]

            batch_mask = mask[start_idx:end_idx]

            batch_x = cx[start_idx:end_idx]

            batch_x_masked = batch_x[batch_mask]

            output_x.append(batch_x_masked)
            output_seq_lens.append(len(batch_x_masked))

        output_x = torch.cat(output_x, dim=0)

        return PackedPoint(coord=output_x, seq_lens=output_seq_lens)

    def assign_batches(
        self,
        bs: T.Union[T.List[int], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        return_clone: bool = False,
    ):
        """
        Assign batches based on their indices.

        Args:
            bs:
                (i, ) List of batch indices.
            x:
                (ni + nj + ..., d) A packed tensor that matches the packing of the indexed batches in the PackedPoint.
            y:
                (n1 + n2 + ... + nb, d) A packed tensor that matches the packing of the current PackedPoint.

        Returns:
            (n1 + n2 + ... + nb, d) New tensor with batches replaced.

        """
        start_idx = self.batch_start_idxs[bs]
        end_idx = start_idx + self.seq_lens[bs]
        ranges = zip(start_idx, end_idx)

        output = y.clone() if return_clone else y
        indices = torch.cat([torch.arange(start, end, device=x.device) for start, end in ranges])
        output[indices, :] = x

        return output

    def get_unpack_idxs(self, max_size: int = None):
        """
        Get the index along the first dimension, ie, (n1 + n2 + ... + nb)
        when unpacking into a padded tensor.

        Args:
            max_size:
                max number of points to keep.
                if ni is less than max_size, pad with zero.
                If None, max_size = max(seq_lens)

        Returns:
            unpack_idx:
                (b, max_size) long, padded with 0
            valid_mask:
                (b, max_size) bool
        """
        if max_size is None:
            max_size = torch.max(self.seq_lens).item()

        # create unpack index
        device = self.device
        batch_indices = torch.arange(self.batch_size, device=device).unsqueeze(1).expand(-1, max_size)  # (b, m)
        seq_indices = torch.arange(max_size, device=device).unsqueeze(0).expand(self.batch_size, -1)  # (b, m)
        valid_mask = seq_indices < torch.clamp(self.seq_lens, max=max_size).unsqueeze(1)  # (b, m)

        # Create flat indices into the packed tensor
        packed_indices = self.batch_start_idxs[:-1].unsqueeze(1) + seq_indices  # (b, m)

        # Only use indices where mask is True
        b_indices = batch_indices[valid_mask]  # (total_valid,)
        s_indices = seq_indices[valid_mask]  # (total_valid,)
        p_indices = packed_indices[valid_mask]  # (total_valid,)

        unpack_idx = torch.zeros(self.batch_size, max_size, dtype=torch.long, device=device)  # (b, m)
        unpack_idx[b_indices, s_indices] = p_indices

        return dict(
            unpack_idx=unpack_idx,  # (b, max_size)
            valid_mask=valid_mask,  # (b, max_size)
        )

    def unpack_arr(
        self,
        packed_arr: T.Union[T.List[torch.Tensor], torch.Tensor],
        max_size: int = None,
    ):
        """
        Unpack a packed_arr corresponding to the packed point.

        Args:
            packed_arr:
                (n1+n2+..+nb, *d) or list of packed_arr to be unpacked
            max_size:
                max number of points to keep.
                if ni is less than max_size, pad with zero.
                If None, max_size = max(seq_lens)

        Returns:
            unpacked_arr:
                (b, max_size, *d) or list of (b, max_size, *d)
        """
        if max_size is None:
            max_size = torch.max(self.seq_lens).item()
        first_idxs = self.batch_start_idxs[:-1]  # (b,)
        assert (first_idxs < self.bn).all(), f"{first_idxs} >= {self.bn}"

        if isinstance(packed_arr, torch.Tensor):
            packed_arr = [packed_arr]
            return_tensor = True
        else:
            return_tensor = False

        # create unpack index
        unpack_idx_dict = self.get_unpack_idxs(max_size=max_size)
        unpack_idx = unpack_idx_dict["unpack_idx"]  # (b, max_size)
        valid_mask = unpack_idx_dict["valid_mask"]  # (b, max_size)
        b, m = unpack_idx.shape

        out_arr = []
        for arr in packed_arr:
            bn, *d_shape = arr.shape
            out = arr[unpack_idx.reshape(-1)].reshape(b, m, *d_shape)
            out[~valid_mask] = 0
            out_arr.append(out)

        if return_tensor:
            out_arr = out_arr[0]

        return dict(
            unpacked_arr=out_arr,  # (b, max_size, *d) or list of it
            valid_mask=valid_mask,  # (b, max_size)
            unpack_idx=unpack_idx,  # (b, max_size)
        )

    def get_bijk_info(
        self,
        cell_width: T.Union[float, torch.Tensor],
        shift: T.Union[float, torch.Tensor],
        save_to_cache: bool,
        attn_backend: str = ATTN_BACKEND,
        printout: bool = False,
    ) -> T.Dict[str, T.Any]:
        """
        Get the linear index, attn_biases, forward and backward sort_idxs, etc
        for voxelization with cell_width and shift.

        This is useful in voxel downsampling and other voxelization operations.

        Args:
            cell_width:
                float or (dn,), cell width of the voxels
            shift:
                float or (dn,), coordinate shift before voxelization
            save_to_cache:
                whether to save the output to cache
            printout:
                whether to print the bijk info (number of points in a cell, etc)

        Returns:
            linear_idx:
                (n1+n2+...+nb,) long, cell index unique to bijk, from [0, total_cells-1].
                The order of the elements are before sorted by forward_idxs.
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
            chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into 0..n1+n2+...+nb)

            if attn_backend == 'xformers':
                attn_biases:
                    (num_chunks,) list of attn_bias uses by xformer

            if attn_backend == 'flash':

                cu_seq_lens:
                    list of (num_chunks,), each is of length (num_cells_in_the_chunk+1,) int32,
                    which contains the cumsum of seq_len in the chunk starts from 0
                max_seq_lens:
                    (num_chunks,)
                    max of seq_len in each chunk

        """

        cache_name = f"bijk_cw{cell_width}_shift{shift}_backend{attn_backend}"
        info_dict = self.get_cache(name=cache_name)
        if info_dict is not None:
            return info_dict

        # get bijk
        stime = timer()
        bidxs = self.batch_idxs  # (bn,) long
        cumprod_grid_size = None
        if self.coord_lim is None:
            # we use floor so it is closer to what is usually expected, eg, 512^3 in [-1, 1]
            ijk = torch.floor((self.coord + shift) / cell_width).long()  # (bn, dn) long
            # we want to make i (ie, x) moves fastest to follow the convention of pytoch (d, h, w)
            # as i for x (w), j for y (h), k for z (d).
            bijk = torch.cat(
                [
                    bidxs.unsqueeze(-1),  # (bn, 1)
                    torch.flip(ijk, dims=[-1]),  # (bn, dn), flip so i moves fastest during unique
                ],
                dim=-1,
            )  # (bn, 1+dn) long  bkji
        else:
            eps = 1e-8
            grid_size = torch.ceil((self.coord_lim[..., 1] - eps - self.coord_lim[..., 0]) / cell_width)  # (dn,) ijk
            cumprod_grid_size = torch.cumprod(grid_size, dim=-1)  # (dn,)  ijk
            coord = (
                torch.minimum(
                    torch.maximum(self.coord, self.coord_lim[..., 0]),  # (bn, dn)
                    self.coord_lim[..., 1] - eps,  # (dn,)
                )
                + shift
            )  # (bn, dn), clip coordinates into the range of self.coord_lim
            ijk = torch.floor((coord - self.coord_lim[..., 0]) / cell_width).long()  # (bn, dn) long
            # flat indices: i + N_x * j + N_x * N_y * k as cumprod_grid_size is [N_x, N_x * N_y, N_x * N_y * N_z].
            # Thus, i/x moves fastest
            ijk[..., 1:] = ijk[..., 1:] * cumprod_grid_size[:-1]
            bijk = ijk.sum(dim=-1) + bidxs * cumprod_grid_size[-1]  # (bn,)  long, i moves fastest, b slowest
            del ijk

        # sort and count by unique
        # with unique, the first dimension moves slowest
        cell_bijk, linear_idx, num_points_in_cells = torch.unique(
            bijk,  # (bn, 1+dn) long  or (bn,) long
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )  # cell_bijk: (total_cells, 1+dn) or (total_cells,)  linear_idx: (bn,)  num_points_in_cells: (total_cells,)
        del bijk  # bijk has double meaning, so delete to prevent misuse
        total_cells = num_points_in_cells.size(0)
        forward_idxs = torch.argsort(linear_idx, dim=0)  # (bn,)
        backward_idxs = torch.empty_like(forward_idxs)  # (bn,)
        backward_idxs.scatter_(0, forward_idxs, torch.arange(len(forward_idxs), device=forward_idxs.device))  # (bn,)

        # get cell_bidx
        if cell_bijk.ndim > 1:
            cell_bidx = cell_bijk[:, 0]  # (total_cells,)
        else:
            assert cumprod_grid_size is not None
            cell_bidx = cell_bijk // cumprod_grid_size[-1]  # (total_cells,)
        assert (cell_bidx >= 0).all(), f"{torch.min(cell_bidx)=}"
        assert (cell_bidx < self.batch_size).all(), f"{torch.max(cell_bidx)=}, {self.batch_size=}"
        assert (cell_bidx[:-1] <= cell_bidx[1:]).all(), f"not sorted: {cell_bidx=}"

        # get new batch count for voxel downsampling, i.e., number of occupied cells per batch
        new_seq_lens = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)  # (b,)
        new_seq_lens.scatter_reduce_(
            dim=0,
            index=cell_bidx,  # (total_cells,)
            src=torch.ones_like(cell_bidx),  # (total_cells,)
            reduce="sum",
            include_self=False,
        )

        # the max number of blocks supported by flash attention is 65536
        if attn_backend == "xformers":
            # num_points_in_cells_list = num_points_in_cells.tolist()
            chunk_size = 30000  # 16384  # 65535
            num_chunks = (total_cells + chunk_size - 1) // chunk_size
            # print(f'num_chunks={num_chunks}')
            # create attn_bias
            attn_biases = []
            chunk_start_idxs = []
            current_idx = 0
            for chunk_idx in range(num_chunks):
                # print(f'min num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]: {min(num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size])}')
                # print(
                #     f'max num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]: {max(num_points_in_cells_list[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])}')
                # print(f'current_idx: {current_idx}')

                # # debug
                # q_seqlen = num_points_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                # assert (q_seqlen > 0).all()
                # assert current_idx + q_seqlen.sum() <= self.bn, f'{current_idx + q_seqlen.sum()}, {current_idx}, {q_seqlen.sum()}, {self.bn}'
                # # end debug

                # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
                #     q_seqlen=num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                # )
                attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                    q_seqlen=num_points_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                )
                attn_biases.append(attn_bias)
                chunk_start_idxs.append(current_idx)
                current_idx = current_idx + attn_bias.q_seqinfo.seqstart_py[-1]
            assert current_idx == self.bn
            chunk_start_idxs.append(self.bn)

            info_dict = dict(
                linear_idx=linear_idx,  # (n1+n2+...+nb,) long
                new_seq_lens=new_seq_lens,  # (b,), only used for voxel_downsampling
                cell_counts=num_points_in_cells,  # (total_cells,)
                total_cells=total_cells,  # int
                forward_idxs=forward_idxs,  # (n1+n2+...+nb,)
                backward_idxs=backward_idxs,  # (n1+n2+...+nb,)
                attn_biases=attn_biases,  # (num_chunks,)
                chunk_start_idxs=chunk_start_idxs,  # (num_chunks+1,)
            )

        elif attn_backend in ("flash", "pytorch"):
            # construct cu_seq_lens and max_seq_lens
            num_points_in_cells = num_points_in_cells.to(dtype=torch.int32)

            # we have checked flash attention support seq_len = 0
            # the max number of blocks supported by flash attention is 65536
            chunk_size = 30000  # 65535  # 30000  # 65535
            num_chunks = (total_cells + chunk_size - 1) // chunk_size
            cu_seq_lens = []
            max_seq_lens = []
            chunk_start_idxs = []
            current_idx = 0

            for chunk_idx in range(num_chunks):
                _seq_len = num_points_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]  # (chunk_size,)
                _cu_seq_len = torch.cat(
                    [
                        torch.zeros(1, dtype=_seq_len.dtype, device=_seq_len.device),
                        torch.cumsum(_seq_len, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )  # (chunk_size+1,)
                _max_seq_len = torch.max(_seq_len).item()
                cu_seq_lens.append(_cu_seq_len)
                max_seq_lens.append(_max_seq_len)
                chunk_start_idxs.append(current_idx)
                current_idx = current_idx + _cu_seq_len[-1]
            chunk_start_idxs.append(self.bn)
            assert current_idx == self.bn

            info_dict = dict(
                linear_idx=linear_idx,  # (n1+n2+...+nb,) long
                new_seq_lens=new_seq_lens,  # (b,), only used for voxel_downsampling
                cell_counts=num_points_in_cells,  # (total_cells,)
                total_cells=total_cells,  # int
                forward_idxs=forward_idxs,  # (n1+n2+...+nb,)
                backward_idxs=backward_idxs,  # (n1+n2+...+nb,)
                chunk_start_idxs=chunk_start_idxs,  # (num_chunks+1,)
                cu_seq_lens=cu_seq_lens,  # (num_chunks,)
                max_seq_lens=max_seq_lens,  # (num_chunks,)
            )
        else:
            raise NotImplementedError(attn_backend)

        if save_to_cache:
            self.insert_cache(name=cache_name, content=info_dict)

        total_time = timer() - stime

        if printout:
            print(
                f"bijk_info for cell width: {cell_width}, shift: {shift}:\n"
                f"  total time: {total_time} secs \n"
                f"  total_cells: {info_dict['total_cells']} \n"
                f"  avg in a cell: {info_dict['cell_counts'].float().mean().item()} \n"
                f"  std in a cell: {info_dict['cell_counts'].float().std().item()} \n"
                f"  min in a cell: {info_dict['cell_counts'].min().item()} \n"
                f"  max in a cell: {info_dict['cell_counts'].max().item()}  "
            )
        return info_dict

    def get_b_query_info(
        self,
        num_query: int,
        save_to_cache: bool,
    ):
        r"""
        Get the attn_biases, chunk_start_idxs, etc for global cross attention
        from a dense tensor (b, num_query, d) to the corresponding
        samples in the packed_point.

        Args:
            num_query:
                number of query in each batch
            save_to_cache:
                whether to save the output to cache

        Returns:
            attn_biases:
                (num_chunks,) list of attn_bias uses by xformer
            kv_chunk_start_idxs:
                (num_chunks+1,) index of coord (ie, into 0..n1+n2+...+nb)
        """
        cache_name = f"bquery_nq{num_query}"
        info_dict = self.get_cache(name=cache_name)
        if info_dict is not None:
            return info_dict

        stime = timer()
        b = self.batch_size
        q_seq_lens = [num_query] * b  # (b,)
        kv_seq_lens = self.seq_lens  # (b,)

        # the max number of blocks supported by flash attention is 65536
        chunk_size = 30000  # 16384  # 65535
        num_chunks = (b + chunk_size - 1) // chunk_size
        # create attn_bias
        attn_biases = []
        kv_chunk_start_idxs = []
        current_idx = 0
        for chunk_idx in range(num_chunks):
            # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            #     q_seqlen=q_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
            #     kv_seqlen=(kv_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]).tolist(),
            # )
            attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                q_seqlen=q_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                kv_seqlen=kv_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size].tolist(),
            )
            attn_biases.append(attn_bias)
            kv_chunk_start_idxs.append(current_idx)
            current_idx = current_idx + attn_bias.k_seqinfo.seqstart_py[-1]
        kv_chunk_start_idxs.append(self.bn)

        info_dict = dict(
            chunk_size=chunk_size,
            attn_biases=attn_biases,  # (num_chunks,)
            kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,)
        )
        if save_to_cache:
            self.insert_cache(name=cache_name, content=info_dict)

        total_time = timer() - stime
        return info_dict

    def get_b_packed_query_info(
        self,
        query_seq_lens: torch.Tensor,
        save_to_cache: bool,
        attn_backend: str = ATTN_BACKEND,
    ):
        r"""
        Get the attn_biases, chunk_start_idxs, etc for global cross attention
        from a packed query (m1+m2+...+mb, d) to the corresponding
        samples in the packed_point.

        Args:
            query_seq_lens:
                (b,), ie, [m1, m2, ..., mb]
            save_to_cache:
                whether to save the output to cache
            attn_backend:
                'xformers'
                'flash'

        Returns:
            kv_chunk_start_idxs:
                (num_chunks+1,) index of coord (ie, into 0..n1+n2+...+nb)

            if attn_backend == 'xformers':
                attn_biases:
                    (num_chunks,) list of attn_bias uses by xformer

            if attn_backend == 'flash':
                q_cu_seq_lens:
                    (num_chunks,), list of (chunk_size+1,)
                q_max_seq_lens:
                    (num_chunks,), list of int
                kv_cu_seq_lens:
                    (num_chunks,), list of (chunk_size+1,)
                kv_max_seq_lens:
                    (num_chunks,), list of int

        """
        if isinstance(query_seq_lens, torch.Tensor):
            query_seq_lens_list = query_seq_lens.tolist()  # (b,)
        else:
            query_seq_lens_list = query_seq_lens

        cache_name = f"b_packed_query_{query_seq_lens_list}"
        info_dict = self.get_cache(name=cache_name)
        if info_dict is not None:
            return info_dict

        stimer = timer()
        b = self.batch_size

        if attn_backend == "xformers":
            # q_seq_lens = query_seq_lens_list  # (b,)
            # kv_seq_lens = self.seq_lens.tolist()  # (b,)
            q_seq_lens = query_seq_lens
            kv_seq_lens = self.seq_lens

            # the max number of blocks supported by flash attention is 65536
            chunk_size = 30000  # 16384  # 65535
            num_chunks = (b + chunk_size - 1) // chunk_size
            # create attn_bias
            attn_biases = []
            q_chunk_start_idxs = []
            q_current_idx = 0
            kv_chunk_start_idxs = []
            kv_current_idx = 0
            for chunk_idx in range(num_chunks):
                # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
                #     q_seqlen=q_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                #     kv_seqlen=kv_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                # )
                attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                    q_seqlen=q_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                    kv_seqlen=kv_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                )
                attn_biases.append(attn_bias)
                q_chunk_start_idxs.append(q_current_idx)
                q_current_idx = q_current_idx + attn_bias.q_seqinfo.seqstart_py[-1]
                kv_chunk_start_idxs.append(kv_current_idx)
                kv_current_idx = kv_current_idx + attn_bias.k_seqinfo.seqstart_py[-1]
            q_chunk_start_idxs.append(q_current_idx)
            kv_chunk_start_idxs.append(kv_current_idx)
            assert kv_current_idx == self.bn, f"{kv_current_idx} != {self.bn}"

            info_dict = dict(
                chunk_size=chunk_size,
                attn_biases=attn_biases,  # (num_chunks,)
                q_chunk_start_idxs=q_chunk_start_idxs,  # (num_chunks+1,)
                kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,)
            )
        elif attn_backend == "flash":
            if isinstance(query_seq_lens, torch.Tensor):
                query_seq_lens = query_seq_lens.to(dtype=torch.int32)
            elif isinstance(query_seq_lens, (list, tuple)):
                query_seq_lens = torch.tensor(query_seq_lens, dtype=torch.int32, device=self.device)
            else:
                raise RuntimeError(type(query_seq_lens))

            q_seq_lens = query_seq_lens.to(dtype=torch.int32)  # (b,)
            kv_seq_lens = self.seq_lens.to(dtype=torch.int32)  # (b,)

            # the max number of blocks supported by flash attention is 65536
            chunk_size = 30000  # 65535
            num_chunks = (b + chunk_size - 1) // chunk_size

            q_chunk_start_idxs = []
            q_cu_seq_lens = []
            q_max_seq_lens = []
            q_current_idx = 0
            kv_chunk_start_idxs = []
            kv_cu_seq_lens = []
            kv_max_seq_lens = []
            kv_current_idx = 0
            for chunk_idx in range(num_chunks):
                _q_seq_lens = q_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                _kv_seq_lens = kv_seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

                _q_cu_seq_lens = torch.cat(
                    [
                        torch.zeros(1, dtype=_q_seq_lens.dtype, device=_q_seq_lens.device),
                        torch.cumsum(_q_seq_lens, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                _kv_cu_seq_lens = torch.cat(
                    [
                        torch.zeros(1, dtype=_kv_seq_lens.dtype, device=_kv_seq_lens.device),
                        torch.cumsum(_kv_seq_lens, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                _q_max_seq_lens = torch.max(_q_seq_lens).item()
                _kv_max_seq_lens = torch.max(_kv_seq_lens).item()

                q_cu_seq_lens.append(_q_cu_seq_lens)
                kv_cu_seq_lens.append(_kv_cu_seq_lens)
                q_max_seq_lens.append(_q_max_seq_lens)
                kv_max_seq_lens.append(_kv_max_seq_lens)
                q_chunk_start_idxs.append(q_current_idx)
                q_current_idx = q_current_idx + _q_cu_seq_lens[-1]
                kv_chunk_start_idxs.append(kv_current_idx)
                kv_current_idx = kv_current_idx + _kv_cu_seq_lens[-1]
            q_chunk_start_idxs.append(q_current_idx)
            kv_chunk_start_idxs.append(kv_current_idx)
            assert kv_current_idx == self.bn, f"{kv_current_idx} != {self.bn}"

            info_dict = dict(
                chunk_size=chunk_size,
                q_cu_seq_lens=q_cu_seq_lens,  # (num_chunks,)
                q_max_seq_lens=q_max_seq_lens,  # (num_chunks,)
                kv_cu_seq_lens=kv_cu_seq_lens,  # (num_chunks,)
                kv_max_seq_lens=kv_max_seq_lens,  # (num_chunks,)
                q_chunk_start_idxs=q_chunk_start_idxs,  # (num_chunks+1,)
                kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,)
            )
        else:
            raise NotImplementedError(attn_backend)

        if save_to_cache:
            self.insert_cache(name=cache_name, content=info_dict)
        total_time = timer() - stimer
        return info_dict

    def get_localized_cross_knn_info(
        self,
        coord_query: "PackedPoint",  # (bn, dn)
        cache_name: str = None,
        use_cached: bool = False,
        save_to_cache: bool = False,
        attn_backend: str = ATTN_BACKEND,
        printout: bool = False,
    ):
        """
        Get the information to perform cross attention that treats
        each query as a cluster center, assign key/value to the
        nearest cluster/query, and finally query only attends to the key/value
        belonging to its cluster.

        Args:
            coord_query:
                (bn, dn) packed
            attn_backend:
                'xformers'
                'flash'

        Returns:
            forward_idxs:
                (bm,)  index to sort the points before performing block attention.
                To use forward_idx, first reshape(b, m, *d) to (bm, *d) then
                do arr[forward_idxs].
            # backward_idxs: (no need to sort back since we only sort key/value)
            #     (bm,) index to sort the points back after performing block attention
            #     To use, do arr[backward_idxs] then reshape back to (b, m, *d)
            attn_biases:
                (num_chunks,) list of attn_bias uses by xformer
            q_chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into bn)
            kv_chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into bm)

        Notes:
            1. We tested pytorch3d's speed of doing knn with batch or without batch (for loop),
            it seems with many points (e.g, n1=100k, n2=8k) there is no significant speed difference.

        """
        if use_cached:
            assert cache_name is not None
            info_dict = self.get_cache(name=cache_name)
            if info_dict is not None:
                return info_dict

        stime = timer()
        coord_key = self
        bn = coord_query.bn
        bm = coord_key.bn
        b = coord_query.batch_size

        # 1-knn (note that if coord/ref_coord is in low-precision format (eg, bfloat16)
        # we might have a cluster that contains no points, ie, seq_len = 0)
        kidxs = []  # index in packed
        key_current_idx = 0
        query_current_idx = 0
        with _nullcontext():
            for ib in range(b):
                knn_out = pytorch3d.ops.knn_points(
                    p1=coord_key.coord[key_current_idx : key_current_idx + coord_key.seq_lens[ib]]
                    .unsqueeze(0)
                    .float(),  # (1, seq_len_key[ib], dn)  kv
                    p2=coord_query.coord[query_current_idx : query_current_idx + coord_query.seq_lens[ib]]
                    .unsqueeze(0)
                    .float(),  # (1, seq_len_query[ib], dn) query
                    K=1,
                )
                _kidxs = knn_out.idx.squeeze(-1).squeeze(
                    0
                )  # (seq_len_key[ib],) long, the query index each key belong to
                _kidxs = _kidxs + query_current_idx  # (seq_len_key[ib],), idx in packed
                kidxs.append(_kidxs)
                query_current_idx = query_current_idx + coord_query.seq_lens[ib]  # might make it a tensor
                key_current_idx = key_current_idx + coord_key.seq_lens[ib]  # might make it a tensor
                del knn_out
        kidxs = torch.cat(kidxs, dim=0) if len(kidxs) > 1 else kidxs[0]  # (bm,)
        assert kidxs.shape == (bm,), f"{kidxs.shape} != {bm}"

        # count seq len for each latent,
        # i.e., for each element in query, how many key/value corresponding to its cluster
        seq_lens = torch.zeros(bn, dtype=torch.long, device=kidxs.device)  # (bn)
        seq_lens.scatter_reduce_(
            dim=0,
            index=kidxs,  # (bm,)
            src=torch.ones_like(kidxs),  # (bm,)
            reduce="sum",
            include_self=False,
        )  # (bn,)

        # simply sort by kidx
        forward_idxs = torch.argsort(
            kidxs, dim=0, descending=False
        )  # (bm,) index to sort the points before performing block attention

        # backward_idxs = torch.empty_like(forward_idxs)  # (bm)
        # backward_idxs.scatter_(
        #     dim=0,
        #     index=forward_idxs,  # (bm)
        #     src=torch.arange(forward_idxs.size(0), device=forward_idxs.device),  # (bm,)
        # )  # (bm,) index to sort the points back after performing block attention

        if attn_backend == "xformers":
            # construct attn bias
            # we have checked flash attention support seq_len = 0
            # the max number of blocks supported by flash attention is 65536
            # seq_lens_list = seq_lens.tolist()  # (bn,)
            chunk_size = 30000  # 16384  # 65535
            num_chunks = (bn + chunk_size - 1) // chunk_size
            attn_biases = []
            q_chunk_start_idxs = []
            kv_chunk_start_idxs = []
            current_idx = 0
            for chunk_idx in range(num_chunks):
                # kv_seqlen = seq_lens_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
                #     q_seqlen=[1] * len(kv_seqlen),
                #     kv_seqlen=kv_seqlen,
                # )
                kv_seqlen = seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                    q_seqlen=[1] * len(kv_seqlen),  # each seq_len in kv_seqlen denotes a cluster for a query
                    kv_seqlen=kv_seqlen,
                )
                attn_biases.append(attn_bias)
                q_chunk_start_idxs.append(chunk_idx * chunk_size)
                kv_chunk_start_idxs.append(current_idx)
                current_idx = current_idx + attn_bias.k_seqinfo.seqstart_py[-1]
            q_chunk_start_idxs.append(bn)
            assert current_idx == bm, f"{current_idx=}, {bm=}"
            kv_chunk_start_idxs.append(bm)

            knn_info = dict(
                chunk_size=chunk_size,
                forward_idxs=forward_idxs,  # (bm,)
                # backward_idxs=backward_idxs,  # (bm,)
                attn_biases=attn_biases,  # (num_chunks,)
                q_chunk_start_idxs=q_chunk_start_idxs,  # (num_chunks+1,), index of bn
                kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,), index of bm
            )
        elif attn_backend == "flash":
            seq_lens = seq_lens.to(dtype=torch.int32)

            # we have checked flash attention support seq_len = 0
            # the max number of blocks supported by flash attention is 65536
            chunk_size = 30000  # 65535
            num_chunks = (bn + chunk_size - 1) // chunk_size

            q_chunk_start_idxs = []
            q_cu_seq_lens = []
            q_max_seq_lens = []
            q_current_idx = 0
            kv_chunk_start_idxs = []
            kv_cu_seq_lens = []
            kv_max_seq_lens = []
            kv_current_idx = 0
            for chunk_idx in range(num_chunks):
                _kv_seq_lens = seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]  # (chunk_size,)
                _q_seq_lens = torch.ones_like(_kv_seq_lens)  # (chunk_size,)

                _q_cu_seq_lens = torch.arange(
                    len(_q_seq_lens) + 1, dtype=torch.int32, device=_q_seq_lens.device
                )  # (chunk_size+1,)
                _kv_cu_seq_lens = torch.cat(
                    [
                        torch.zeros(1, dtype=_kv_seq_lens.dtype, device=_kv_seq_lens.device),
                        torch.cumsum(_kv_seq_lens, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                _q_max_seq_lens = 1
                _kv_max_seq_lens = torch.max(_kv_seq_lens).item()

                q_cu_seq_lens.append(_q_cu_seq_lens)
                kv_cu_seq_lens.append(_kv_cu_seq_lens)
                q_max_seq_lens.append(_q_max_seq_lens)
                kv_max_seq_lens.append(_kv_max_seq_lens)
                q_chunk_start_idxs.append(q_current_idx)
                q_current_idx = q_current_idx + _q_cu_seq_lens[-1]
                kv_chunk_start_idxs.append(kv_current_idx)
                kv_current_idx = kv_current_idx + _kv_cu_seq_lens[-1]
            q_chunk_start_idxs.append(q_current_idx)
            kv_chunk_start_idxs.append(kv_current_idx)
            assert kv_current_idx == self.bn, f"{kv_current_idx} != {self.bn}"
            assert q_current_idx == coord_query.bn, f"{q_current_idx} != {coord_query.bn}"

            knn_info = dict(
                chunk_size=chunk_size,
                forward_idxs=forward_idxs,  # (bm,)
                # backward_idxs=backward_idxs,  # (bm,)
                q_cu_seq_lens=q_cu_seq_lens,
                q_max_seq_lens=q_max_seq_lens,
                kv_cu_seq_lens=kv_cu_seq_lens,
                kv_max_seq_lens=kv_max_seq_lens,
                q_chunk_start_idxs=q_chunk_start_idxs,  # (num_chunks+1,), index of bn
                kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,), index of bm
            )
        else:
            raise NotImplementedError(attn_backend)

        if save_to_cache:
            assert cache_name is not None
            self.insert_cache(name=cache_name, content=knn_info)

        total_time = timer() - stime
        if printout:
            print(
                f"knn cross attention:\n"
                f"  total time: {total_time} secs \n"
                f"  total_clusters (b * n): {bn} \n"
                f"  avg in a cluster: {seq_lens.float().mean().item()} \n"
                f"  std in a cluster: {seq_lens.float().std().item()} \n"
                f"  min in a cluster: {seq_lens.min().item()} \n"
                f"  max in a cluster: {seq_lens.max().item()}  "
            )
        return knn_info

    def get_voxel_windowed_cross_knn_info(
        self,
        coord_query: "PackedPoint",  # (bn, dn)
        cell_width: T.Union[float, torch.Tensor],
        shift: T.Union[float, torch.Tensor] = None,
        cache_name: str = None,
        use_cached: bool = False,
        save_to_cache: bool = False,
        attn_backend: str = ATTN_BACKEND,
        printout: bool = False,
    ):
        """
        Get the information to perform cross attention that uses
        voxel to determine the key/value a query can attend to.

        Args:
            coord_query:
                (bn, dn) packed
            cell_width:
                float or (dn,)  those key/value in the same voxel
                as the query will be attended by the query.
            shift:
                float or (dn,)
            attn_backend:
                'xformers'
                'flash'

        Returns:
            forward_idxs:
                (bm,)  index to sort the points before performing block attention.
                To use forward_idx, first reshape(b, m, *d) to (bm, *d) then
                do arr[forward_idxs].
            # backward_idxs: (no need to sort back since we only sort key/value)
            #     (bm,) index to sort the points back after performing block attention
            #     To use, do arr[backward_idxs] then reshape back to (b, m, *d)
            q_chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into bn)
            kv_chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into bm)

            if attn_backend == 'xformers':
                attn_biases:
                    (num_chunks,) list of attn_bias uses by xformer

            if attn_backend == 'flash':
                q_cu_seq_lens:
                    list of (num_chunks,), each is of length (num_cells_in_the_chunk+1,),
                    which contains the cumsum of seq_len in the chunk starts from 0
                q_max_seq_lens:
                    (num_chunks,)
                    max of seq_len in each chunk
                kv_cu_seq_lens:
                    list of (num_chunks,), each is of length (num_cells_in_the_chunk+1,),
                    which contains the cumsum of seq_len in the chunk starts from 0
                kv_max_seq_lens:
                    (num_chunks,)
                    max of seq_len in each chunk


        Notes:
            1. if chose to save the cache, the cache will be saved in key (ie, self).
        """
        if use_cached:
            assert cache_name is not None
            info_dict = self.get_cache(name=cache_name)
            if info_dict is not None:
                return info_dict

        if printout and torch.cuda.is_available():
            torch.cuda.synchronize()

        stime_overall = timer()
        coord_key = self
        bn = coord_query.bn
        bm = coord_key.bn
        b = coord_query.batch_size

        if (coord_query.coord_lim is not None) or (coord_key.coord_lim is not None):
            assert torch.allclose(
                coord_query.coord_lim,
                coord_key.coord_lim,
            )
            coord_lim = coord_query.coord_lim
        else:
            coord_lim = None

        # voxelization
        if coord_lim is None:
            # we use floor so it is closer to what is usually expected, eg, 512^3 in [-1, 1]
            ijk_key = torch.floor((coord_key.coord + shift) / cell_width).long()  # (bm, dn) long
            ijk_query = torch.floor((coord_query.coord + shift) / cell_width).long()  # (bn, dn) long
        else:
            eps = 1e-8
            coord = (
                torch.minimum(
                    torch.maximum(coord_key.coord, coord_key.coord_lim[..., 0]),  # (bm, dn)
                    coord_lim[..., 1] - eps,  # (dn,)
                )
                + shift
            )  # (bm, dn), clip coord_key.coord into the range of coord_key.coord_lim
            ijk_key = torch.floor((coord - coord_lim[..., 0]) / cell_width).long()  # (bm, dn) long
            coord = (
                torch.minimum(
                    torch.maximum(coord_query.coord, coord_query.coord_lim[..., 0]),  # (bn, dn)
                    coord_lim[..., 1] - eps,  # (dn,)
                )
                + shift
            )  # (bm, dn), clip coord_query.coord into the range of coord_query.coord_lim
            ijk_query = torch.floor((coord - coord_lim[..., 0]) / cell_width).long()  # (bn, dn) long

        bijk_key = torch.cat(
            [
                coord_key.batch_idxs.unsqueeze(-1),  # (bm, 1)
                torch.flip(ijk_key, dims=[-1]),  # (bm, dn), flip so i moves fastest during unique
            ],
            dim=-1,
        )  # (bm, 1+dn) long  bkji
        bijk_query = torch.cat(
            [
                coord_query.batch_idxs.unsqueeze(-1),  # (bn, 1)
                torch.flip(ijk_query, dims=[-1]),  # (bn, dn), flip so i moves fastest during unique
            ],
            dim=-1,
        )  # (bn, 1+dn) long  bkji

        bijk = torch.cat([bijk_key, bijk_query], dim=0)  # (bm+bn, 1+dn) bkji

        # sort and count by unique
        stime_unique = timer()
        cell_bijk, linear_idx, num_points_in_cells = torch.unique(
            bijk,  # (bm+bn, 1+dn)
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )  # cell_bijk: (total_cells, 1+dn) or (total_cells,)  linear_idx: (bm+bn,)  num_points_in_cells: (total_cells,)
        del bijk  # bijk has double meaning, so delete to prevent misuse
        total_time_unique = timer() - stime_unique

        # note there might be some voxels that only contain key or only contain query
        # flash attention seems ok to deal with this scenario
        total_cells = num_points_in_cells.size(0)  # include both key and query
        linear_idx_key = linear_idx[:bm]  # (bm,)
        linear_idx_query = linear_idx[bm:]  # (bn,)

        # since we sort key and query separately, we need to compile the forward_idxs separately
        stime_sort_key = timer()
        forward_idxs_key = torch.argsort(linear_idx_key, dim=0)  # (bm,)
        backward_idxs_key = torch.empty_like(forward_idxs_key)  # (bm,)
        backward_idxs_key.scatter_(
            0, forward_idxs_key, torch.arange(len(forward_idxs_key), device=forward_idxs_key.device)
        )  # (bm,)
        total_time_sort_key = timer() - stime_sort_key

        stime_sort_query = timer()
        forward_idxs_query = torch.argsort(linear_idx_query, dim=0)  # (bn,)
        backward_idxs_query = torch.empty_like(forward_idxs_query)  # (bn,)
        backward_idxs_query.scatter_(
            0, forward_idxs_query, torch.arange(len(forward_idxs_query), device=forward_idxs_query.device)
        )  # (bn,)
        total_time_sort_query = timer() - stime_sort_query

        # when constructing attention bias of xformers, we need to count
        # how many tokens/points for query and for key in each voxel separately
        stime_count_query = timer()
        num_query_in_cells = torch.zeros_like(num_points_in_cells)  # (total_cells,)
        num_query_in_cells.scatter_reduce_(
            dim=0,
            index=linear_idx_query,  # (bn,)
            src=torch.ones_like(linear_idx_query),  # (bn,)
            reduce="sum",
            include_self=False,
        )  # (total_cells,)
        total_time_count_query = timer() - stime_count_query

        stime_count_key = timer()
        num_key_in_cells = torch.zeros_like(num_points_in_cells)  # (total_cells,)
        num_key_in_cells.scatter_reduce_(
            dim=0,
            index=linear_idx_key,  # (bm,)
            src=torch.ones_like(linear_idx_key),  # (bm,)
            reduce="sum",
            include_self=False,
        )  # (total_cells,)
        total_time_count_key = timer() - stime_count_key

        # assert torch.allclose(num_query_in_cells + num_key_in_cells, num_points_in_cells)
        del num_points_in_cells

        if attn_backend == "xformers":
            # create attn_bias
            # the max number of blocks supported by flash attention is 65536
            stime_xformer = timer()
            # num_query_in_cells_list = num_query_in_cells.tolist()  # (total_cells,)
            # num_key_in_cells_list = num_key_in_cells.tolist()  # (total_cells,)
            chunk_size = 30000  # 16384  # 65535
            num_chunks = (total_cells + chunk_size - 1) // chunk_size
            attn_biases = []
            q_chunk_start_idxs = []
            q_current_idx = 0
            kv_chunk_start_idxs = []
            kv_current_idx = 0
            for chunk_idx in range(num_chunks):
                # print(f'min num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]: {min(num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size])}')
                # print(
                #     f'max num_points_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]: {max(num_points_in_cells_list[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])}')
                # print(f'current_idx: {current_idx}')

                # # debug
                # q_seqlen = num_points_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                # assert (q_seqlen > 0).all()
                # assert current_idx + q_seqlen.sum() <= self.bn, f'{current_idx + q_seqlen.sum()}, {current_idx}, {q_seqlen.sum()}, {self.bn}'
                # # end debug

                # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
                #     q_seqlen=num_query_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                #     kv_seqlen=num_key_in_cells_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                # )
                attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                    q_seqlen=num_query_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                    kv_seqlen=num_key_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                )
                attn_biases.append(attn_bias)
                q_chunk_start_idxs.append(q_current_idx)
                q_current_idx = q_current_idx + attn_bias.q_seqinfo.seqstart_py[-1]
                kv_chunk_start_idxs.append(kv_current_idx)
                kv_current_idx = kv_current_idx + attn_bias.k_seqinfo.seqstart_py[-1]
            assert q_current_idx == coord_query.bn, f"{q_current_idx=}, {coord_query.bn=}"
            q_chunk_start_idxs.append(coord_query.bn)
            assert kv_current_idx == coord_key.bn, f"{kv_current_idx=}, {coord_key.bn=}"
            kv_chunk_start_idxs.append(coord_key.bn)
            total_time_xformer = timer() - stime_xformer

            info_dict = dict(
                linear_idx_query=linear_idx_query,  # (n1+n2+...+nb,) long
                linear_idx_key=linear_idx_key,  # (m1+m2+...+mb,) long
                total_cells=total_cells,  # int
                forward_idxs_query=forward_idxs_query,  # (n1+n2+...+nb,)
                forward_idxs_key=forward_idxs_key,  # (m1+m2+...+mb,)
                backward_idxs_query=backward_idxs_query,  # (n1+n2+...+nb,)
                backward_idxs_key=backward_idxs_key,  # (m1+m2+...+mb,)
                attn_biases=attn_biases,  # (num_chunks,)
                q_chunk_start_idxs=q_chunk_start_idxs,  # (num_chunks+1,)
                kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,)
            )

        elif attn_backend == "flash":
            num_query_in_cells = num_query_in_cells.to(dtype=torch.int32)
            num_key_in_cells = num_key_in_cells.to(dtype=torch.int32)

            # the max number of blocks supported by flash attention is 65536
            stime_xformer = timer()
            chunk_size = 30000  # 65535
            num_chunks = (total_cells + chunk_size - 1) // chunk_size

            q_chunk_start_idxs = []
            q_cu_seq_lens = []
            q_max_seq_lens = []
            q_current_idx = 0
            kv_chunk_start_idxs = []
            kv_cu_seq_lens = []
            kv_max_seq_lens = []
            kv_current_idx = 0

            for chunk_idx in range(num_chunks):
                _q_seqlens = num_query_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                _kv_seqlens = num_key_in_cells[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                _q_cu_seq_lens = torch.cat(
                    [
                        torch.zeros(1, dtype=_q_seqlens.dtype, device=_q_seqlens.device),
                        torch.cumsum(_q_seqlens, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                _q_max_seq_len = torch.max(_q_seqlens).item()
                _kv_cu_seq_lens = torch.cat(
                    [
                        torch.zeros(1, dtype=_kv_seqlens.dtype, device=_kv_seqlens.device),
                        torch.cumsum(_kv_seqlens, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                _kv_max_seq_len = torch.max(_kv_seqlens).item()

                q_cu_seq_lens.append(_q_cu_seq_lens)
                q_max_seq_lens.append(_q_max_seq_len)
                kv_cu_seq_lens.append(_kv_cu_seq_lens)
                kv_max_seq_lens.append(_kv_max_seq_len)

                q_chunk_start_idxs.append(q_current_idx)
                q_current_idx = q_current_idx + _q_cu_seq_lens[-1]
                kv_chunk_start_idxs.append(kv_current_idx)
                kv_current_idx = kv_current_idx + _kv_cu_seq_lens[-1]

            assert q_current_idx == coord_query.bn
            q_chunk_start_idxs.append(coord_query.bn)
            assert kv_current_idx == coord_key.bn
            kv_chunk_start_idxs.append(coord_key.bn)
            total_time_xformer = timer() - stime_xformer

            info_dict = dict(
                linear_idx_query=linear_idx_query,  # (n1+n2+...+nb,) long
                linear_idx_key=linear_idx_key,  # (m1+m2+...+mb,) long
                total_cells=total_cells,  # int
                forward_idxs_query=forward_idxs_query,  # (n1+n2+...+nb,)
                forward_idxs_key=forward_idxs_key,  # (m1+m2+...+mb,)
                backward_idxs_query=backward_idxs_query,  # (n1+n2+...+nb,)
                backward_idxs_key=backward_idxs_key,  # (m1+m2+...+mb,)
                q_cu_seq_lens=q_cu_seq_lens,  # (num_chunks,)
                q_max_seq_lens=q_max_seq_lens,  # (num_chunks,)
                kv_cu_seq_lens=kv_cu_seq_lens,  # (num_chunks,)
                kv_max_seq_lens=kv_max_seq_lens,  # (num_chunks,)
                q_chunk_start_idxs=q_chunk_start_idxs,  # (num_chunks+1,)
                kv_chunk_start_idxs=kv_chunk_start_idxs,  # (num_chunks+1,)
            )

        else:
            raise NotImplementedError(attn_backend)

        if save_to_cache:
            assert cache_name is not None
            self.insert_cache(name=cache_name, content=info_dict)

        if printout and torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = timer() - stime_overall
        if printout:
            print(
                f"voxel windowed cross for cell width: {cell_width}, shift: {shift}:\n"
                f"  total time: {total_time} secs \n"
                f"  total_time_unique: {total_time_unique} secs \n"
                f"  total_time_sort_key: {total_time_sort_key} secs \n"
                f"  total_time_sort_query: {total_time_sort_query} secs \n"
                f"  total_time_count_key: {total_time_count_key} secs \n"
                f"  total_time_count_query: {total_time_count_query} secs \n"
                f"  total_time_xformer: {total_time_xformer} secs \n"
                f"  total_cells: {info_dict['total_cells']} \n"
                f"  avg query in a cell: {num_query_in_cells.float().mean().item()} \n"
                f"  std query in a cell: {num_query_in_cells.float().std().item()} \n"
                f"  min query in a cell: {num_query_in_cells.min().item()} \n"
                f"  max query in a cell: {num_query_in_cells.max().item()} \n"
                f"  avg key in a cell: {num_key_in_cells.float().mean().item()} \n"
                f"  std key in a cell: {num_key_in_cells.float().std().item()} \n"
                f"  min key in a cell: {num_key_in_cells.min().item()} \n"
                f"  max key in a cell: {num_key_in_cells.max().item()}  "
            )

        return info_dict

    def get_localized_self_knn_info(
        self,
        k: T.Union[int, T.List[int]],
        cache_name: str = None,
        use_cached: bool = False,
        save_to_cache: bool = False,
        attn_backend: str = ATTN_BACKEND,
        debug: bool = False,
    ):
        """
        Get the information to perform self attention that
        randomly determines k clusters and attends to those
        within the same cluster.

        Args:
            k:
                int or list of (b,), number of clusters
            attn_backend:
                'flash'
                'xformers'

        Returns:
            forward_idxs:
                (bn,)  index to sort the points before performing block attention.
                To use forward_idx, first reshape(b, n, *d) to (bn, *d) then
                do arr[forward_idxs].
            backward_idxs: (no need to sort back since we only sort key/value)
                (bn,) index to sort the points back after performing block attention
                To use, do arr[backward_idxs] then reshape back to (b, n, *d)
            q_chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into bn)
            kv_chunk_start_idxs:
                (num_chunks+1,) index of sorted coord (ie, into bm)

            if attn_backend == 'xformers':
                attn_biases:
                    (num_chunks,) list of attn_bias uses by xformer

            if attn_backend == 'flash':

                cu_seq_lens:
                    list of (num_chunks,), each is of length (num_cells_in_the_chunk+1,),
                    which contains the cumsum of seq_len in the chunk starts from 0
                max_seq_lens:
                    (num_chunks,)
                    max of seq_len in each chunk

        Notes:
            1. We tested pytorch3d's speed of doing knn with batch or without batch (for loop),
            it seems with many points (n1=100k, n2=8k) there is no significant speed difference.

        """
        if use_cached:
            assert cache_name is not None
            info_dict = self.get_cache(name=cache_name)
            if info_dict is not None:
                return info_dict

        stime = timer()
        b = self.batch_size
        if isinstance(k, int):
            k = [k] * b
        total_k = sum(k)

        kidxs = []  # index in packed
        current_idx = 0
        current_total_k = 0
        with _nullcontext():
            for ib in range(b):
                _coord = self.coord[current_idx : current_idx + self.seq_lens[ib]]  # (ni, dn)

                if k[ib] >= 32:
                    # randomly determine k clusters
                    ridxs = torch.randperm(_coord.size(0), device=self.coord.device)[: k[ib]]
                else:
                    # too few clusters, let's use farthest point sampling to select the cluster ref
                    with torch.no_grad():
                        # first select a small set (but not too small)
                        _ridxs = torch.randperm(_coord.size(0), device=self.coord.device)[:1024]
                        _, ridxs = pytorch3d.ops.sample_farthest_points(
                            points=_coord[_ridxs].unsqueeze(0),  # (1, nn=1024, dn)
                            K=k[ib],
                            random_start_point=True,
                        )  # (1, k, dn), (1, k)
                        ridxs = _ridxs[ridxs.squeeze(0)]  # (k,)

                _coord_ref = _coord[ridxs]  # (k, dn)

                # run 1-knn
                knn_out = pytorch3d.ops.knn_points(
                    p1=_coord.unsqueeze(0).float(),  # (1, seq_len_key[ib], dn)  kv
                    p2=_coord_ref.unsqueeze(0).float(),  # (1, k, dn) query
                    K=1,
                )
                _kidxs = knn_out.idx.squeeze(-1).squeeze(
                    0
                )  # (seq_len_key[ib],) long, the cluster index each key belong to
                _kidxs = _kidxs + current_total_k
                kidxs.append(_kidxs)
                current_idx = current_idx + self.seq_lens[ib]  # might make it a tensor
                current_total_k = current_total_k + k[ib]
                del knn_out
        kidxs = torch.cat(kidxs, dim=0) if len(kidxs) > 1 else kidxs[0]  # (bn,)
        assert kidxs.shape == (self.bn,), f"{kidxs.shape} != {self.bn}"
        assert current_total_k == total_k, f"{current_total_k} != {total_k}"

        # count seq len for each cluster
        seq_lens = torch.zeros(total_k, dtype=torch.long, device=kidxs.device)  # (bk,)
        seq_lens.scatter_reduce_(
            dim=0,
            index=kidxs,  # (bn,)
            src=torch.ones_like(kidxs),  # (bn,)
            reduce="sum",
            include_self=False,
        )  # (bk,)

        # simply sort by kidx
        forward_idxs = torch.argsort(
            kidxs, dim=0, descending=False
        )  # (bn,) index to sort the points before performing block attention

        backward_idxs = torch.empty_like(forward_idxs)  # (bn,)
        backward_idxs.scatter_(
            dim=0,
            index=forward_idxs,  # (bn,)
            src=torch.arange(forward_idxs.size(0), device=forward_idxs.device),  # (bn,)
        )  # (bn,) index to sort the points back after performing block attention

        if attn_backend == "xformers":
            # construct attn bias
            # we have checked flash attention support seq_len = 0
            # the max number of blocks supported by flash attention is 65536
            # seq_lens_list = seq_lens.tolist()  # (bk,)
            chunk_size = 30000  # 16384  # 65535
            num_chunks = (total_k + chunk_size - 1) // chunk_size
            attn_biases = []
            chunk_start_idxs = []
            current_idx = 0
            for chunk_idx in range(num_chunks):
                attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                    q_seqlen=seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                )
                attn_biases.append(attn_bias)
                chunk_start_idxs.append(current_idx)
                current_idx = current_idx + attn_bias.q_seqinfo.seqstart_py[-1]
            chunk_start_idxs.append(self.bn)
            assert current_idx == self.bn, f"{current_idx=}, {self.bn=}"

            knn_info = dict(
                chunk_size=chunk_size,
                forward_idxs=forward_idxs,  # (bn,)
                backward_idxs=backward_idxs,  # (bn,)
                attn_biases=attn_biases,  # (num_chunks,)
                chunk_start_idxs=chunk_start_idxs,  # (num_chunks+1,), index of bn
                total_k=total_k,  # int, sum of number of cluster across samples in a batch
            )
        elif attn_backend == "flash":
            # construct cu_seq_lens and max_seq_lens

            seq_lens = seq_lens.to(dtype=torch.int32)

            # we have checked flash attention support seq_len = 0
            # the max number of blocks supported by flash attention is 65536
            chunk_size = 30000  # 65535
            num_chunks = (total_k + chunk_size - 1) // chunk_size
            cu_seq_lens = []
            max_seq_lens = []
            chunk_start_idxs = []
            current_idx = 0

            for chunk_idx in range(num_chunks):
                _seq_len = seq_lens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]  # (chunk_size,)
                _cu_seq_len = torch.cat(
                    [
                        torch.zeros(1, dtype=_seq_len.dtype, device=_seq_len.device),
                        torch.cumsum(_seq_len, dim=0, dtype=torch.int32),  # (chunk_size,)
                    ],
                    dim=0,
                )  # (chunk_size+1,)
                _max_seq_len = torch.max(_seq_len).item()
                cu_seq_lens.append(_cu_seq_len)
                max_seq_lens.append(_max_seq_len)
                chunk_start_idxs.append(current_idx)
                current_idx = current_idx + _cu_seq_len[-1]
            chunk_start_idxs.append(self.bn)
            assert current_idx == self.bn

            knn_info = dict(
                chunk_size=chunk_size,
                forward_idxs=forward_idxs,  # (bn,)
                backward_idxs=backward_idxs,  # (bn,)
                cu_seq_lens=cu_seq_lens,  # (num_chunks,)
                max_seq_lens=max_seq_lens,  # (num_chunks,)
                chunk_start_idxs=chunk_start_idxs,  # (num_chunks+1,), index of bn
                total_k=total_k,  # int, sum of number of cluster across samples in a batch
            )
        else:
            raise NotImplementedError(attn_backend)

        if debug:
            knn_info["kidxs"] = kidxs  # (bn,)

        if save_to_cache:
            assert cache_name is not None
            self.insert_cache(name=cache_name, content=knn_info)

        total_time = timer() - stime
        if debug:
            print(
                f"knn self attention (k = {k}):\n"
                f"  total time: {total_time} secs \n"
                f"  total_clusters (b * k): {total_k} \n"
                f"  avg in a cluster: {seq_lens.float().mean().item()} \n"
                f"  std in a cluster: {seq_lens.float().std().item()} \n"
                f"  min in a cluster: {seq_lens.min().item()} \n"
                f"  max in a cluster: {seq_lens.max().item()}  "
            )
        return knn_info


def voxel_downsampling(
    packed_coord: "PackedPoint",
    packed_feature: torch.Tensor,
    cell_width: T.Union[float, torch.Tensor],
    shift: float,
    save_to_cache: bool,
    aggregation_method: str = "mean",
    printout: bool = False,
) -> T.Dict[str, T.Union[torch.Tensor, "PackedPoint"]]:
    r"""
    Voxel downsampling by averaging the coordinate and feature of points within a cell.

    Procedure:
    - Points are discretized into voxels.
    - Each occupied voxel generates exactly one point by
      averaging all points inside

    Args:
        packed_coord:
            (n1+n2+...+nb, dn)
        packed_feature:
            (n1+n2+...+nb, *d)
        cell_width:
            the width of each grid cell.
            If <0, return self (do nothing)
        aggregation_method:
            'mean': average the coord in a voxel
            'subsample': randomly select one in a voxel
        save_to_cache:
            whether to save the index in the cache of packed_coord

    Returns:
        packed_coord:
            (num_occupied_cells, dn)
        packed_feature:
            (num_occupied_cells, *d)
    """

    if aggregation_method == "mean":
        return voxel_downsampling_avg(
            packed_coord=packed_coord,
            packed_feature=packed_feature,
            cell_width=cell_width,
            shift=shift,
            save_to_cache=save_to_cache,
            printout=printout,
        )
    elif aggregation_method == "subsample":
        return voxel_subsampling(
            packed_coord=packed_coord,
            packed_feature=packed_feature,
            cell_width=cell_width,
            shift=shift,
            save_to_cache=save_to_cache,
            printout=printout,
        )
    else:
        raise NotImplementedError


def voxel_downsampling_avg(
    packed_coord: "PackedPoint",
    packed_feature: torch.Tensor,
    cell_width: T.Union[float, torch.Tensor],
    shift: float,
    save_to_cache: bool,
    printout: bool = False,
) -> T.Dict[str, T.Union[torch.Tensor, "PackedPoint"]]:
    r"""
    Voxel downsampling by averaging the coordinate and feature of points within a cell.

    Procedure:
    - Points are discretized into voxels.
    - Each occupied voxel generates exactly one point by
      averaging all points inside

    Args:
        packed_coord:
            (n1+n2+...+nb, dn)
        packed_feature:
            (n1+n2+...+nb, *d)
        cell_width:
            the width of each grid cell.
            If <0, return self (do nothing)
        save_to_cache:
            whether to save the index in the cache of packed_coord

    Returns:
        packed_coord:
            (num_occupied_cells, dn)
        packed_feature:
            (num_occupied_cells, *d)
    """

    # get bijk info (ie, assign points to cells)
    bijk_info_dict = packed_coord.get_bijk_info(
        cell_width=cell_width,
        shift=shift,
        save_to_cache=save_to_cache,
        printout=printout,
    )
    idxs = bijk_info_dict["linear_idx"]  # (n1+n2+..+nb,) long, individual point -> occupied cell index
    counts = bijk_info_dict["cell_counts"]  # (num_cells,) long
    num_occupied_cells = counts.size(0)
    new_seq_lens = bijk_info_dict["new_seq_lens"]  # (b,)

    # average xyz to get new xyz (so implicitly weighted by sample density)
    dn = packed_coord.dn
    coords_mean = torch.zeros(
        num_occupied_cells, dn, dtype=packed_coord.dtype, device=packed_coord.device
    )  # (num_occupied_cells, dn)
    coords_mean.scatter_reduce_(
        dim=0,
        index=idxs.unsqueeze(-1).expand(-1, dn),  # (bn, dn)
        src=packed_coord.coord,  # (bn, dn)
        reduce="mean",
        include_self=False,  # important, do not want to include 0 and the count
    )

    # average xyz to get new xyz (so implicitly weighted by sample density)
    bn, *d_shape = packed_feature.shape
    assert bn == packed_coord.bn, f"{bn=}, {packed_coord.bn=}"
    d = math.prod(d_shape)
    packed_feature = packed_feature.view(bn, d)  # (bn, d)

    feat_mean = torch.zeros(
        num_occupied_cells, d, dtype=packed_feature.dtype, device=packed_feature.device
    )  # (num_occupied_cells, d)
    feat_mean.scatter_reduce_(
        dim=0,
        index=idxs.unsqueeze(-1).expand(-1, d),  # (bn, d)
        src=packed_feature,  # (bn, d)
        reduce="mean",
        include_self=False,  # important, do not want to include 0 and the count
    )  # (num_occupied_cells, d)
    # reshape d back to *d
    feat_mean = feat_mean.view(num_occupied_cells, *d_shape)  # (num_occupied_cells, *d)

    new_packed_coord = PackedPoint(
        coord=coords_mean,  # (num_occupied_cells, dn)
        seq_lens=new_seq_lens,  # (b,)
        coord_lim=packed_coord.coord_lim,  # (dn, 2) or None
    )

    return dict(
        packed_coord=new_packed_coord,  # (num_occupied_cells, dn)
        packed_feature=feat_mean,  # (num_occupied_cells, *d)
    )


def voxel_subsampling(
    packed_coord: "PackedPoint",
    packed_feature: torch.Tensor,
    cell_width: T.Union[float, torch.Tensor],
    shift: float,
    save_to_cache: bool,
    printout: bool = False,
) -> T.Dict[str, T.Union[torch.Tensor, "PackedPoint"]]:
    r"""
    Voxel downsampling by subsampling the coordinate and feature of points within a cell.

    Procedure:
    - Points are discretized into voxels.
    - Each occupied voxel generates exactly one point by
      randomly select the points inside

    Args:
        packed_coord:
            (n1+n2+...+nb, dn)
        packed_feature:
            (n1+n2+...+nb, *d)
        cell_width:
            the width of each grid cell.
            If <0, return self (do nothing)
        save_to_cache:
            whether to save the index in the cache of packed_coord

    Returns:
        packed_coord:
            (num_occupied_cells, dn)
        packed_feature:
            (num_occupied_cells, *d)
    """

    # get bijk info (ie, assign points to cells)
    bijk_info_dict = packed_coord.get_bijk_info(
        cell_width=cell_width,
        shift=shift,
        save_to_cache=save_to_cache,
        printout=printout,
    )
    counts = bijk_info_dict["cell_counts"]  # (num_cells,) long
    forward_idxs = bijk_info_dict["forward_idxs"]  # (n1+n2+..+nb,) long
    num_occupied_cells = counts.size(0)
    new_seq_lens = bijk_info_dict["new_seq_lens"]  # (b,)

    # sort
    coord = packed_coord.coord[forward_idxs]  # (bn, dn)
    feature = packed_feature[forward_idxs]  # (bn, *d)

    # select one from each cell
    ridxs = torch.floor(
        torch.rand(num_occupied_cells, dtype=coord.dtype, device=coord.device) * counts
    )  # (num_cells,) [0, num_in_cell-1] long
    offset = torch.cat(
        [
            torch.zeros_like(counts[0:1]),  # (1,)
            torch.cumsum(counts, dim=0, dtype=torch.int32),  # (num_cells,)
        ],
        dim=0,
    )  # (num_cells + 1)
    ridxs = ridxs + offset[:-1]  # (num_cells,)

    new_coord = coord[ridxs]  # (num_cells, dn)
    new_feature = feature[ridxs]  # (num_cells, *d)
    new_packed_coord = PackedPoint(
        coord=new_coord,  # (num_occupied_cells, dn)
        seq_lens=new_seq_lens,  # (b,)
        coord_lim=packed_coord.coord_lim,  # (dn, 2) or None
    )

    return dict(
        packed_coord=new_packed_coord,  # (num_occupied_cells, dn)
        packed_feature=new_feature,  # (num_occupied_cells, *d)
    )


def voxel_windowed_self_softmax_attention(
    packed_coord: "PackedPoint",
    packed_query: torch.Tensor,
    packed_key: torch.Tensor,
    packed_value: torch.Tensor,
    cell_width: T.Union[float, torch.Tensor],
    shift: float,
    save_to_cache: bool,
    printout: bool = False,
) -> torch.Tensor:
    r"""
    Assign the points into individual cells, perform self-attention among the points within the same cell.

    Args:
        packed_coord:
            (n1 + n2 + ... + nb, dn) coordinate of the query and key (since self attention, they have the same coord)
        packed_query:
            (n1 + n2 + ... + nb, h, d)
        packed_key:
            (n1 + n2 + ... + nb, h, d)
        packed_value:
            (n1 + n2 + ... + nb, h, d)
        cell_width:
            float or (dn,) the width of each cell
        shift:
            shift the origin to allow mixing between the cells
        save_to_cache:
            whether to save the assignment and attention bias to packed_coord's cache

    Returns:
        out:
        (n1 + n2 + ... + nb, h, d), the result of self-attention

    Notes:
        We will use the xformers.ops.fmha.attn_bias.BlockDiagonalMask to perform
        self attention within each cell.

        To use it, we will first sort the points based on their batch_idx and cell_ijk.
        We will use unique on the index (4bijk,) or (5bijkl). This gives us cell counts as well.

    Notes:
        In order to use flash attention, we need
            device={'cuda'}
            dtype={torch.float16, torch.bfloat16}
            query.shape[-1] % 8 == 0

    Complexity:
        The computational complexity is O(m * k^2), where m is the number of nonempty cells, and
        k is the number of points in each cell -- oversimplified calculation.

    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0, f"{packed_query.shape=}"
    assert packed_key.size(-1) % 8 == 0, f"{packed_key.shape=}"
    assert packed_value.size(-1) % 8 == 0, f"{packed_value.shape=}"
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    # get bijk
    bijk_dict = packed_coord.get_bijk_info(
        cell_width=cell_width,
        shift=shift,
        save_to_cache=save_to_cache,
        printout=printout,
        attn_backend="xformers",
    )
    forward_idxs = bijk_dict["forward_idxs"]  # (bn,)
    backward_idxs = bijk_dict["backward_idxs"]  # (bn,)
    attn_biases = bijk_dict["attn_biases"]  # (num_chunks)
    chunk_start_idxs = bijk_dict["chunk_start_idxs"]  # (num_chunks+1,)

    # run xformer attention
    # sort feature
    packed_query = packed_query[forward_idxs]  # (n1 + n2 + ... + nb, h, d)
    packed_key = packed_key[forward_idxs]  # (n1 + n2 + ... + nb, h, d)
    packed_value = packed_value[forward_idxs]  # (n1 + n2 + ... + nb, h, d)

    with _nullcontext():
        outs = []
        for chunk_idx in range(len(attn_biases)):
            attn_bias = attn_biases[chunk_idx]

            # print(f'actual len: {len(range(chunk_start_idxs[chunk_idx], chunk_start_idxs[chunk_idx + 1]))}')
            assert ((attn_bias.q_seqinfo.seqstart[1:] - attn_bias.q_seqinfo.seqstart[:-1]) > 0).all()
            out = xops.memory_efficient_attention(
                query=packed_query[
                    None, chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]
                ],  # (b=1, n, h, d)
                key=packed_key[None, chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]],  # (b=1, n, h, d)
                value=packed_value[
                    None, chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]
                ],  # (b=1, n, h, d)
                attn_bias=attn_bias,
            )  # .to(dtype=ori_value_dtype)  # (b=1, n, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        out = out.squeeze(0)  # (bn, h, d)
        # sort back
        out = out[backward_idxs]
        out = out.to(dtype=ori_value_dtype)
    return out


def voxel_windowed_self_softmax_attention_flash_stacked(
    packed_coord: "PackedPoint",
    packed_qkv: torch.Tensor,
    cell_width: T.Union[float, torch.Tensor],
    shift: float,
    save_to_cache: bool,
    printout: bool = False,
) -> torch.Tensor:
    r"""
    Assign the points into individual cells, perform self-attention among the points within the same cell.

    Args:
        packed_coord:
            (n1 + n2 + ... + nb, dn) coordinate of the query and key (since self attention, they have the same coord)
        packed_qkv:
            (n1 + n2 + ... + nb, 3qkv, h, d)
        cell_width:
            float or (dn,) the width of each cell
        shift:
            shift the origin to allow mixing between the cells
        save_to_cache:
            whether to save the assignment and attention bias to packed_coord's cache

    Returns:
        out:
        (n1 + n2 + ... + nb, h, d), the result of self-attention

    Notes:
        We will use the flash_attn_varlen_qkvpacked_func to perform
        self attention within each cell.

        To use it, we will first sort the points based on their batch_idx and cell_ijk.
        We will use unique on the index (4bijk,) or (5bijkl). This gives us cell counts as well.

    Notes:
        In order to use flash attention, we need
            device={'cuda'}
            dtype={torch.float16, torch.bfloat16}
            query.shape[-1] % 8 == 0

    Complexity:
        The computational complexity is O(m * k^2), where m is the number of nonempty cells, and
        k is the number of points in each cell -- oversimplified calculation.

    """

    # prepare for flash attention
    assert packed_qkv.size(-1) % 8 == 0
    ori_value_dtype = packed_qkv.dtype
    if packed_qkv.dtype not in (torch.float16, torch.bfloat16):
        packed_qkv = packed_qkv.to(torch.bfloat16)

    # get bijk
    bijk_dict = packed_coord.get_bijk_info(
        cell_width=cell_width,
        shift=shift,
        save_to_cache=save_to_cache,
        attn_backend="flash",
        printout=printout,
    )
    forward_idxs = bijk_dict["forward_idxs"]  # (bn,)
    backward_idxs = bijk_dict["backward_idxs"]  # (bn,)
    cu_seq_lens = bijk_dict["cu_seq_lens"]  # list (num_chunks,)
    max_seq_lens = bijk_dict["max_seq_lens"]  # list (num_chunks,)
    chunk_start_idxs = bijk_dict["chunk_start_idxs"]  # (num_chunks+1,)

    # run attention
    # sort feature
    packed_qkv = packed_qkv[forward_idxs]  # (n1 + n2 + ... + nb, 3qkv, h, d)
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(cu_seq_lens)):
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                packed_qkv[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]],  # (bn', 3qkv, h, d)
                cu_seq_lens[chunk_idx],  # (num_cells + 1,)
                max_seq_lens[chunk_idx],  # (num_cells + 1,)
            )  # (bn', h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bn, h, d)
        # sort back
        out = out[backward_idxs]
        out = out.to(dtype=ori_value_dtype)
    return out


def voxel_windowed_self_softmax_attention_flash(
    packed_coord: "PackedPoint",
    packed_query: torch.Tensor,
    packed_key: torch.Tensor,
    packed_value: torch.Tensor,
    cell_width: T.Union[float, torch.Tensor],
    shift: float,
    save_to_cache: bool,
    printout: bool = False,
) -> torch.Tensor:
    r"""
    Assign the points into individual cells, perform self-attention among the points within the same cell.

    Args:
        packed_coord:
            (n1 + n2 + ... + nb, dn) coordinate of the query and key (since self attention, they have the same coord)
        packed_query:
            (n1 + n2 + ... + nb, h, d)
        packed_key:
            (n1 + n2 + ... + nb, h, d)
        packed_value:
            (n1 + n2 + ... + nb, h, d)
        cell_width:
            float or (dn,) the width of each cell
        shift:
            shift the origin to allow mixing between the cells
        save_to_cache:
            whether to save the assignment and attention bias to packed_coord's cache

    Returns:
        out:
        (n1 + n2 + ... + nb, h, d), the result of self-attention

    Notes:
        We will use the flash_attn_varlen_func to perform
        self attention within each cell.

        To use it, we will first sort the points based on their batch_idx and cell_ijk.
        We will use unique on the index (4bijk,) or (5bijkl). This gives us cell counts as well.

    Notes:
        In order to use flash attention, we need
            device={'cuda'}
            dtype={torch.float16, torch.bfloat16}
            query.shape[-1] % 8 == 0

    Complexity:
        The computational complexity is O(m * k^2), where m is the number of nonempty cells, and
        k is the number of points in each cell -- oversimplified calculation.

    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_key.size(-1) % 8 == 0
    assert packed_value.size(-1) % 8 == 0
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_key = packed_key.to(dtype=packed_value.dtype)
    packed_query = packed_query.to(dtype=packed_value.dtype)

    # get bijk
    bijk_dict = packed_coord.get_bijk_info(
        cell_width=cell_width,
        shift=shift,
        save_to_cache=save_to_cache,
        attn_backend="flash",
        printout=printout,
    )
    forward_idxs = bijk_dict["forward_idxs"]  # (bn,)
    backward_idxs = bijk_dict["backward_idxs"]  # (bn,)
    cu_seq_lens = bijk_dict["cu_seq_lens"]  # list (num_chunks,)
    max_seq_lens = bijk_dict["max_seq_lens"]  # list (num_chunks,)
    chunk_start_idxs = bijk_dict["chunk_start_idxs"]  # (num_chunks+1,)

    # run attention
    # sort feature
    packed_query = packed_query[forward_idxs]  # (n1 + n2 + ... + nb, h, d)
    packed_key = packed_key[forward_idxs]  # (n1 + n2 + ... + nb, h, d)
    packed_value = packed_value[forward_idxs]  # (n1 + n2 + ... + nb, h, d)
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(cu_seq_lens)):
            out = flash_attn.flash_attn_varlen_func(
                q=packed_query[
                    chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]
                ].contiguous(),  # (bn', h, d)
                k=packed_key[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]].contiguous(),  # (bn', h, d)
                v=packed_value[
                    chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]
                ].contiguous(),  # (bn', h, d)
                cu_seqlens_q=cu_seq_lens[chunk_idx].int().contiguous(),  # (num_cells + 1,)
                cu_seqlens_k=cu_seq_lens[chunk_idx].int().contiguous(),  # (num_cells + 1,)
                max_seqlen_q=max_seq_lens[chunk_idx],
                max_seqlen_k=max_seq_lens[chunk_idx],
            )  # (bn', h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bn, h, d)
        # sort back
        out = out[backward_idxs]
        out = out.to(dtype=ori_value_dtype)
    return out


def cross_softmax_attention_with_packed_kv(
    query: torch.Tensor,  # (b, m, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    save_to_cache: bool,
) -> torch.Tensor:
    r"""
    Perform cross attention between the query and the keys and values
    from the same batch_idx in packed_key and packed_value.

    Args:
        query:
            (b, m, h, d), h for the number of heads
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        save_to_cache:
            whether to save the attn_bias to packed_kv_coord's cache

    Returns:
        (b, m, h, d) result of the cross attention

    Notes:
        We will use the xformers.ops.fmha.attn_bias.BlockDiagonalMask to perform
        cross attention

        To use it, we will construct the attn_bias using seq_lens.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # NOTE from @xiaoming_zhao3: I do not think this is needed
    # as flash_attention will automatically pad the head_dimension.
    # See https://github.com/Dao-AILab/flash-attention/issues/1347#issuecomment-2489241622
    # prepare for flash attention
    assert query.size(-1) % 8 == 0, f"{query.shape=}"
    assert packed_key.size(-1) % 8 == 0, f"{packed_key.shape=}"
    assert packed_value.size(-1) % 8 == 0, f"{packed_value.shape=}"
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    query = query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    b, m, h, d = query.shape
    dv = packed_value.size(-1)
    assert b == packed_kv_coord.batch_size

    # get b_query info
    b_query_dict = packed_kv_coord.get_b_query_info(
        num_query=m,
        save_to_cache=save_to_cache,
    )
    attn_biases = b_query_dict["attn_biases"]  # (num_chunks,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    chunk_size = b_query_dict["chunk_size"]  # int

    # run xformer attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(attn_biases)):
            _query = query[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size].reshape(
                1, -1, h, d
            )  # (b=1, cm, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            attn_bias = attn_biases[chunk_idx]

            out = xops.memory_efficient_attention(
                query=_query,  # (b=1, cm, h, d)
                key=_key,  # (b=1, n, h, d)
                value=_value,  # (b=1, n, h, d)
                attn_bias=attn_bias,
            )  # (b=1, cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]  # (b=1, bm, h, d)
        out = out.reshape(b, m, h, dv)  # (b, m, h, dv)
        out = out.to(dtype=ori_value_dtype)

    return out


def cross_softmax_attention_with_packed_qkv(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    save_to_cache: bool,
) -> torch.Tensor:
    r"""
    Perform global cross attention between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        save_to_cache:
            whether to save the attn_bias to packed_kv_coord's cache

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use the xformers.ops.fmha.attn_bias.BlockDiagonalMask to perform
        cross attention

        To use it, we will construct the attn_bias using seq_lens.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0, f"{packed_query.shape=}"
    assert packed_key.size(-1) % 8 == 0, f"{packed_key.shape=}"
    assert packed_value.size(-1) % 8 == 0, f"{packed_value.shape=}"
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    bm, h, d = packed_query.shape
    dv = packed_value.size(-1)

    # get b_query info
    b_query_dict = packed_kv_coord.get_b_packed_query_info(
        query_seq_lens=packed_query_coord.seq_lens,
        save_to_cache=save_to_cache,
        attn_backend="xformers",
    )
    attn_biases = b_query_dict["attn_biases"]  # (num_chunks,)
    q_chunk_start_idxs = b_query_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    chunk_size = b_query_dict["chunk_size"]  # int

    # run xformer attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(attn_biases)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, m, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            attn_bias = attn_biases[chunk_idx]

            out = xops.memory_efficient_attention(
                query=_query,  # (b=1, cm, h, d)
                key=_key,  # (b=1, n, h, d)
                value=_value,  # (b=1, n, h, d)
                attn_bias=attn_bias,
            )  # (b=1, cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]  # (b=1, bm, h, d)
        out = out.squeeze(0)  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)

    return out


def cross_softmax_attention_with_packed_qkv_flash_stacked(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_kv: torch.Tensor,  # (n1+n2+...+nb, 2kv, h, d)
    save_to_cache: bool,
) -> torch.Tensor:
    r"""
    Perform global cross attention between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_kv:
            (n1+n2+...+nb, 2kv, h, d)
        save_to_cache:
            whether to save the attn_bias to packed_kv_coord's cache

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use flash_attn_varlen_kvpacked_func to perform
        cross attention.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_kv.size(-1) % 8 == 0
    ori_value_dtype = packed_kv.dtype
    if packed_kv.dtype not in (torch.float16, torch.bfloat16):
        packed_kv = packed_kv.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_kv.dtype)

    bm, h, d = packed_query.shape

    # get b_query info
    b_query_dict = packed_kv_coord.get_b_packed_query_info(
        query_seq_lens=packed_query_coord.seq_lens,
        save_to_cache=save_to_cache,
        attn_backend="flash",
    )
    q_chunk_start_idxs = b_query_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    q_cu_seq_lens = b_query_dict["q_cu_seq_lens"]  # (num_chunks,)
    q_max_seq_lens = b_query_dict["q_max_seq_lens"]  # (num_chunks,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    kv_cu_seq_lens = b_query_dict["kv_cu_seq_lens"]  # (num_chunks,)
    kv_max_seq_lens = b_query_dict["kv_max_seq_lens"]  # (num_chunks,)

    # run attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(q_cu_seq_lens)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]]  # (cm, h, d)
            _key_value = packed_kv[
                kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]
            ]  # (cn, 2kv, h, d)

            out = flash_attn.flash_attn_varlen_kvpacked_func(
                q=_query,  # (cm, h, d)
                kv=_key_value,  # (cn, 2kv, h, d)
                cu_seqlens_q=q_cu_seq_lens[chunk_idx],
                cu_seqlens_k=kv_cu_seq_lens[chunk_idx],
                max_seqlen_q=q_max_seq_lens[chunk_idx],
                max_seqlen_k=kv_max_seq_lens[chunk_idx],
            )  # (cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)
        out = out.to(dtype=ori_value_dtype)

    return out


def cross_softmax_attention_with_packed_qkv_flash(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    save_to_cache: bool,
) -> torch.Tensor:
    r"""
    Perform global cross attention between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        save_to_cache:
            whether to save the attn_bias to packed_kv_coord's cache

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use flash_attn_varlen_kvpacked_func to perform
        cross attention.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_key.size(-1) % 8 == 0
    assert packed_value.size(-1) % 8 == 0
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    bm, h, d = packed_query.shape

    # get b_query info
    b_query_dict = packed_kv_coord.get_b_packed_query_info(
        query_seq_lens=packed_query_coord.seq_lens,
        save_to_cache=save_to_cache,
        attn_backend="flash",
    )
    q_chunk_start_idxs = b_query_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    q_cu_seq_lens = b_query_dict["q_cu_seq_lens"]  # (num_chunks,)
    q_max_seq_lens = b_query_dict["q_max_seq_lens"]  # (num_chunks,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    kv_cu_seq_lens = b_query_dict["kv_cu_seq_lens"]  # (num_chunks,)
    kv_max_seq_lens = b_query_dict["kv_max_seq_lens"]  # (num_chunks,)

    # run attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(q_cu_seq_lens)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]]  # (cm, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]]  # (cn, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]]  # (cn, h, d)

            out = flash_attn.flash_attn_varlen_func(
                q=_query,  # (cm, h, d)
                k=_key,  # (cn, h, d)
                v=_value,  # (cn, h, d)
                cu_seqlens_q=q_cu_seq_lens[chunk_idx],
                cu_seqlens_k=kv_cu_seq_lens[chunk_idx],
                max_seqlen_q=q_max_seq_lens[chunk_idx],
                max_seqlen_k=kv_max_seq_lens[chunk_idx],
            )  # (cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)
        out = out.to(dtype=ori_value_dtype)

    return out


def self_softmax_attention_with_packed_qkv_flash_stacked(
    packed_coord: "PackedPoint",  # (m1+m2+..._mb, dn)
    packed_qkv: torch.Tensor,  # (m1+m2+..._mb, 3qkv, h, d)
) -> torch.Tensor:
    r"""
    Perform global cross attention between the packed query,
    keys and values using flash attention's
    `flash_attn_varlen_qkvpacked_func`.

    Args:
        packed_coord:
            (m1+m2+...mb, dn)
        packed_qkv:
            (bm, 3qkv, h, d) stacked qkv
        save_to_cache:
            whether to save the attn_bias to packed_kv_coord's cache

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_qkv.size(-1) % 8 == 0
    ori_value_dtype = packed_qkv.dtype
    if packed_qkv.dtype not in (torch.float16, torch.bfloat16):
        packed_qkv = packed_qkv.to(torch.bfloat16)

    bm, _3qkv, h, d = packed_qkv.shape
    assert _3qkv == 3, f"{packed_qkv.shape}"
    # flash attention only support varlen segments up to 65536
    # we can implement chunking but since it is batch size,
    # we do not expect we will have more than this number
    assert packed_coord.batch_size <= 65536, f"{packed_coord.batch_size} > 65536"

    # get b_query info
    seq_lens = packed_coord.seq_lens  # (b,)
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=seq_lens.device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ],
        dim=0,
    ).int()  # (b+1,) int32

    # run
    with _nullcontext():
        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            packed_qkv,  # (bm, 3qkv, h, d)
            cu_seqlens,  # (b + 1,)
            seq_lens.max().item(),
        )  # (bm, h, d)
    out = out.to(dtype=ori_value_dtype)  # (bm, h, d)
    return out


def self_softmax_attention_with_packed_qkv_flash(
    packed_coord: "PackedPoint",  # (m1+m2+..._mb, dn)
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_key: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_value: torch.Tensor,  # (m1+m2+..._mb, h, d)
) -> torch.Tensor:
    r"""
    Perform global cross attention between the packed query,
    keys and values using flash attention's
    `flash_attn_varlen_qkvpacked_func`.

    Args:
        packed_coord:
            (m1+m2+...mb, dn)
        packed_qkv:
            (bm, h, d)
        save_to_cache:
            whether to save the attn_bias to packed_kv_coord's cache

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_key.size(-1) % 8 == 0
    assert packed_value.size(-1) % 8 == 0

    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    bm, h, d = packed_query.shape

    # flash attention only support varlen segments up to 65536
    # we can implement chunking but since it is batch size,
    # we do not expect we will have more than this number
    assert packed_coord.batch_size <= 65536, f"{packed_coord.batch_size} > 65536"

    # get b_query info
    seq_lens = packed_coord.seq_lens  # (b,)
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=seq_lens.device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ],
        dim=0,
    ).int()  # (b+1,) int32
    max_seq_len = seq_lens.max().item()

    # run
    with _nullcontext():
        out = flash_attn.flash_attn_varlen_func(
            q=packed_query,  # (bm, h, d)
            k=packed_key,  # (bm, h, d)
            v=packed_value,  # (bm, h, d)
            cu_seqlens_q=cu_seqlens,  # (b + 1,)
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seq_len,
            max_seqlen_k=max_seq_len,
        )  # (bm, h, d)
        out = out.to(dtype=ori_value_dtype)  # (bm, h, d)
    return out


def localized_knn_cross_softmax_attention_with_packed_qkv(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    use_cached: bool = False,
    cache_name: str = None,
    debug: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized knn cross attention between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.


    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use the xformers.ops.fmha.attn_bias.BlockDiagonalMask to perform
        cross attention

        To use it, we will construct the attn_bias using seq_lens.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0, f"{packed_query.shape=}"
    assert packed_key.size(-1) % 8 == 0, f"{packed_query.shape=}"
    assert packed_value.size(-1) % 8 == 0, f"{packed_value.shape=}"
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    # get b_query info
    b_query_dict = packed_kv_coord.get_localized_cross_knn_info(
        coord_query=packed_query_coord,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="xformers",
        printout=debug,
    )
    attn_biases = b_query_dict["attn_biases"]  # (num_chunks,)
    q_chunk_start_idxs = b_query_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    forward_idxs = b_query_dict["forward_idxs"]  # (bn,)

    # no need to sort query, but need to sort kv
    packed_key = packed_key[forward_idxs]  # (bn, h, d)
    packed_value = packed_value[forward_idxs]  # (bn, h, d)

    # run xformer attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(attn_biases)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, m, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            attn_bias = attn_biases[chunk_idx]

            if _key.numel() > 0:
                out = xops.memory_efficient_attention(
                    query=_query,  # (b=1, cm, h, d)
                    key=_key,  # (b=1, n, h, d)
                    value=_value,  # (b=1, n, h, d)
                    attn_bias=attn_bias,
                )  # (b=1, cm, h, d)
            else:
                # _key and _value contain no elements
                out = torch.zeros(
                    *_query.shape[:-1],
                    _value.shape[-1],
                    dtype=_value.dtype,
                    device=_value.device,
                )  # (b=1, cm, h, d), multiply

                # NOTE: this is a must-have as it uses the parameter packed_query for this chunk:
                # - enable gradient backpropagation
                # - make sure DDP's gradients have same shape across devices
                out = out * (_query.mean())
            outs.append(out)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]  # (b=1, bm, h, d)
        out = out.squeeze(0)  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)

        # no need to sort out (since query was not sorted)

    return out


def localized_knn_cross_softmax_attention_with_packed_qkv_flash_stacked(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_kv: torch.Tensor,  # (n1+n2+...+nb, 2kv, h, d)
    use_cached: bool = False,
    cache_name: str = None,
    debug: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized knn cross attention between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_kv:
            (n1+n2+...+nb, 2kv, h, d)
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We use flash_attn_varlen_kvpacked_func to perform
        cross attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_kv.size(-1) % 8 == 0
    ori_value_dtype = packed_kv.dtype
    if packed_kv.dtype not in (torch.float16, torch.bfloat16):
        packed_kv = packed_kv.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_kv.dtype)

    # get b_query info
    b_query_dict = packed_kv_coord.get_localized_cross_knn_info(
        coord_query=packed_query_coord,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="flash",
        printout=debug,
    )

    q_chunk_start_idxs = b_query_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    q_cu_seq_lens = b_query_dict["q_cu_seq_lens"]  # (num_chunks,)
    q_max_seq_lens = b_query_dict["q_max_seq_lens"]  # (num_chunks,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    kv_cu_seq_lens = b_query_dict["kv_cu_seq_lens"]  # (num_chunks,)
    kv_max_seq_lens = b_query_dict["kv_max_seq_lens"]  # (num_chunks,)
    forward_idxs = b_query_dict["forward_idxs"]  # (bn,)

    # no need to sort query, but need to sort kv
    packed_kv = packed_kv[forward_idxs]  # (bn, 2kv, h, d)

    # run attention
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(q_cu_seq_lens)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]]  # (cm, h, d)
            _key_value = packed_kv[
                kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]
            ]  # (cn, 2kv, h, d)

            out = flash_attn.flash_attn_varlen_kvpacked_func(
                q=_query,  # (cm, h, d)
                kv=_key_value,  # (cn, 2kv, h, d)
                cu_seqlens_q=q_cu_seq_lens[chunk_idx],
                cu_seqlens_k=kv_cu_seq_lens[chunk_idx],
                max_seqlen_q=q_max_seq_lens[chunk_idx],
                max_seqlen_k=kv_max_seq_lens[chunk_idx],
            )  # (cm, h, d)
            outs.append(out)

        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)
        out = out.to(dtype=ori_value_dtype)

        # no need to sort out (since query was not sorted)

    return out


def localized_knn_cross_softmax_attention_with_packed_qkv_flash(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    use_cached: bool = False,
    cache_name: str = None,
    debug: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized knn cross attention between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We use flash_attn_varlen_kvpacked_func to perform
        cross attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_key.size(-1) % 8 == 0
    assert packed_value.size(-1) % 8 == 0
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    # get b_query info
    b_query_dict = packed_kv_coord.get_localized_cross_knn_info(
        coord_query=packed_query_coord,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="flash",
        printout=debug,
    )

    q_chunk_start_idxs = b_query_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    q_cu_seq_lens = b_query_dict["q_cu_seq_lens"]  # (num_chunks,)
    q_max_seq_lens = b_query_dict["q_max_seq_lens"]  # (num_chunks,)
    kv_chunk_start_idxs = b_query_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    kv_cu_seq_lens = b_query_dict["kv_cu_seq_lens"]  # (num_chunks,)
    kv_max_seq_lens = b_query_dict["kv_max_seq_lens"]  # (num_chunks,)
    forward_idxs = b_query_dict["forward_idxs"]  # (bn,)

    # no need to sort query, but need to sort kv
    packed_key = packed_key[forward_idxs]  # (bn, 2kv, h, d)
    packed_value = packed_value[forward_idxs]  # (bn, 2kv, h, d)

    # run attention
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(q_cu_seq_lens)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]]  # (cm, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]]  # (cn, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]]  # (cn, h, d)
            out = flash_attn.flash_attn_varlen_func(
                q=_query,  # (cm, h, d)
                k=_key,  # (cn, 2kv, h, d)
                v=_value,  # (cn, 2kv, h, d)
                cu_seqlens_q=q_cu_seq_lens[chunk_idx],
                cu_seqlens_k=kv_cu_seq_lens[chunk_idx],
                max_seqlen_q=q_max_seq_lens[chunk_idx],
                max_seqlen_k=kv_max_seq_lens[chunk_idx],
            )  # (cm, h, d)
            outs.append(out)

        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)
        out = out.to(dtype=ori_value_dtype)

        # no need to sort out (since query was not sorted)

    return out


def voxel_windowed_cross_softmax_attention_with_packed_qkv(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    cell_width: T.Union[float, torch.Tensor],
    shift: T.Union[float, torch.Tensor] = 0,
    use_cached: bool = False,
    cache_name: str = None,
    debug: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized cross attention based on voxel window
    between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        cell_width:
            float or (dn,)
        shift:
            float or (dn,)
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.


    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use the xformers.ops.fmha.attn_bias.BlockDiagonalMask to perform
        cross attention

        To use it, we will construct the attn_bias using seq_lens.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0, f"{packed_query.shape=}"
    assert packed_key.size(-1) % 8 == 0, f"{packed_key.shape=}"
    assert packed_value.size(-1) % 8 == 0, f"{packed_value.shape=}"
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    # check boundary
    if (packed_query_coord.coord_lim is not None) or (packed_kv_coord.coord_lim is not None):
        assert torch.allclose(packed_query_coord.coord_lim, packed_kv_coord.coord_lim), (
            f"{packed_query_coord.coord_lim=}, {packed_kv_coord.coord_lim=}"
        )

    # get attentino bias
    info_dict = packed_kv_coord.get_voxel_windowed_cross_knn_info(
        coord_query=packed_query_coord,
        cell_width=cell_width,
        shift=shift,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="xformers",
        printout=debug,
    )

    attn_biases = info_dict["attn_biases"]  # (num_chunks,)
    q_chunk_start_idxs = info_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    kv_chunk_start_idxs = info_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    forward_idxs_query = info_dict["forward_idxs_query"]  # (bm,)
    backward_idxs_query = info_dict["backward_idxs_query"]  # (bm,)
    forward_idxs_key = info_dict["forward_idxs_key"]  # (bn,)

    # sort
    packed_query = packed_query[forward_idxs_query]  # (bm, h, d)
    packed_key = packed_key[forward_idxs_key]  # (bn, h, d)
    packed_value = packed_value[forward_idxs_key]  # (bn, h, d)

    # run xformer attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(attn_biases)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, m, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            attn_bias = attn_biases[chunk_idx]

            out = xops.memory_efficient_attention(
                query=_query,  # (b=1, cm, h, d)
                key=_key,  # (b=1, n, h, d)
                value=_value,  # (b=1, n, h, d)
                attn_bias=attn_bias,
            )  # (b=1, cm, h, d)
            outs.append(out)

        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]  # (b=1, bm, h, d)
        out = out.squeeze(0)  # (bm, h, dv)
        # sort back
        out = out[backward_idxs_query]  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)
    return out


def voxel_windowed_cross_softmax_attention_with_packed_qkv_flash_stacked(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_kv: torch.Tensor,  # (n1+n2+...+nb, 2kv, h, d)
    cell_width: T.Union[float, torch.Tensor],
    shift: T.Union[float, torch.Tensor] = 0,
    use_cached: bool = False,
    cache_name: str = None,
    debug: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized cross attention based on voxel window
    between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_kv:
            (n1+n2+...+nb, 2kv, h, d)
        cell_width:
            float or (dn,)
        shift:
            float or (dn,)
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use flash_attn_varlen_kvpacked_func to perform
        cross attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_kv.size(-1) % 8 == 0
    ori_value_dtype = packed_kv.dtype
    if packed_kv.dtype not in (torch.float16, torch.bfloat16):
        packed_kv = packed_kv.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_kv.dtype)

    # check boundary
    if packed_query_coord.coord_lim is not None or packed_kv_coord.coord_lim is not None:
        assert torch.allclose(packed_query_coord.coord_lim, packed_kv_coord.coord_lim)

    # get attentino bias
    info_dict = packed_kv_coord.get_voxel_windowed_cross_knn_info(
        coord_query=packed_query_coord,
        cell_width=cell_width,
        shift=shift,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="flash",
        printout=debug,
    )

    q_chunk_start_idxs = info_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    q_cu_seq_lens = info_dict["q_cu_seq_lens"]  # (num_chunks,)
    q_max_seq_lens = info_dict["q_max_seq_lens"]  # (num_chunks,)
    kv_chunk_start_idxs = info_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    kv_cu_seq_lens = info_dict["kv_cu_seq_lens"]  # (num_chunks,)
    kv_max_seq_lens = info_dict["kv_max_seq_lens"]  # (num_chunks,)

    forward_idxs_query = info_dict["forward_idxs_query"]  # (bm,)
    backward_idxs_query = info_dict["backward_idxs_query"]  # (bm,)
    forward_idxs_key = info_dict["forward_idxs_key"]  # (bn,)

    # sort
    packed_query = packed_query[forward_idxs_query]  # (bm, h, d)
    packed_kv = packed_kv[forward_idxs_key]  # (bn, h, d)

    # run attention
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(q_cu_seq_lens)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]]  # (cm, h, d)
            _key_value = packed_kv[
                kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]
            ]  # (cn, 2kv, h, d)

            out = flash_attn.flash_attn_varlen_kvpacked_func(
                q=_query,  # (cm, h, d)
                kv=_key_value,  # (cn, 2kv, h, d)
                cu_seqlens_q=q_cu_seq_lens[chunk_idx],
                cu_seqlens_k=kv_cu_seq_lens[chunk_idx],
                max_seqlen_q=q_max_seq_lens[chunk_idx],
                max_seqlen_k=kv_max_seq_lens[chunk_idx],
            )  # (cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)

        # sort back
        out = out[backward_idxs_query]  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)

    return out


def voxel_windowed_cross_softmax_attention_with_packed_qkv_flash(
    packed_query_coord: "PackedPoint",
    packed_query: torch.Tensor,  # (m1+m2+..._mb, h, d)
    packed_kv_coord: "PackedPoint",  # (n1+n2+...+nb, dn)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    cell_width: T.Union[float, torch.Tensor],
    shift: T.Union[float, torch.Tensor] = 0,
    use_cached: bool = False,
    cache_name: str = None,
    debug: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized cross attention based on voxel window
    between the packed query and the packed keys and packed values.

    Args:
        packed_query_coord:
            (m1+m2+...mb, dn)
        packed_query:
            (bm, h, d)
        packed_kv_coord:
            (n1+n2+...+nb, dn)  packed coordinate for key and value
        packed_key:
            (n1+n2+...+nb, h, d)
        packed_value:
            (n1+n2+...+nb, h, d)
        cell_width:
            float or (dn,)
        shift:
            float or (dn,)
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (m1+m2+..._mb, h, d) result of the cross attention

    Notes:
        We will use flash_attn_varlen_func to perform
        cross attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_key.size(-1) % 8 == 0
    assert packed_value.size(-1) % 8 == 0
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    # check boundary
    if packed_query_coord.coord_lim is not None or packed_kv_coord.coord_lim is not None:
        assert torch.allclose(packed_query_coord.coord_lim, packed_kv_coord.coord_lim)

    # get attentino bias
    info_dict = packed_kv_coord.get_voxel_windowed_cross_knn_info(
        coord_query=packed_query_coord,
        cell_width=cell_width,
        shift=shift,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="flash",
        printout=debug,
    )

    q_chunk_start_idxs = info_dict["q_chunk_start_idxs"]  # (num_chunks+1,)
    q_cu_seq_lens = info_dict["q_cu_seq_lens"]  # (num_chunks,)
    q_max_seq_lens = info_dict["q_max_seq_lens"]  # (num_chunks,)
    kv_chunk_start_idxs = info_dict["kv_chunk_start_idxs"]  # (num_chunks+1,)
    kv_cu_seq_lens = info_dict["kv_cu_seq_lens"]  # (num_chunks,)
    kv_max_seq_lens = info_dict["kv_max_seq_lens"]  # (num_chunks,)

    forward_idxs_query = info_dict["forward_idxs_query"]  # (bm,)
    backward_idxs_query = info_dict["backward_idxs_query"]  # (bm,)
    forward_idxs_key = info_dict["forward_idxs_key"]  # (bn,)

    # sort
    packed_query = packed_query[forward_idxs_query]  # (bm, h, d)
    packed_key = packed_key[forward_idxs_key]  # (bn, h, d)
    packed_value = packed_value[forward_idxs_key]  # (bn, h, d)

    # run attention
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(q_cu_seq_lens)):
            _query = packed_query[q_chunk_start_idxs[chunk_idx] : q_chunk_start_idxs[chunk_idx + 1]]  # (cm, h, d)
            _key = packed_key[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]]  # (cn, h, d)
            _value = packed_value[kv_chunk_start_idxs[chunk_idx] : kv_chunk_start_idxs[chunk_idx + 1]]  # (cn, h, d)

            out = flash_attn.flash_attn_varlen_func(
                q=_query,  # (cm, h, d)
                k=_key,  # (cn, 2kv, h, d)
                v=_value,  # (cn, 2kv, h, d)
                cu_seqlens_q=q_cu_seq_lens[chunk_idx],
                cu_seqlens_k=kv_cu_seq_lens[chunk_idx],
                max_seqlen_q=q_max_seq_lens[chunk_idx],
                max_seqlen_k=kv_max_seq_lens[chunk_idx],
            )  # (cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)

        # sort back
        out = out[backward_idxs_query]  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)

    return out


def localized_knn_self_softmax_attention_packed(
    packed_coord: "PackedPoint",  # (n1+n2+...+nb, h, d)
    packed_query: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    k: T.Union[int, T.List[int]],
    use_cached: bool = False,
    cache_name: str = None,
    printout: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized knn self attention.

    This function is intended for using after QKV fed through linear layers.
    Thus, though this is for self-attention, we have different QKV as input.
    However, QKV share the same coordinates.

    Args:
        packed_coord:
            (n1+n2+...nb, dn)
        packed_query:
            (bn, h, d)
        packed_key:
            (bn, h, d)
        packed_value:
            (bn, h, d)
        k:
            int or list of (b,) number of clusters
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (n1+n2+...+nb, h, d) result of the self attention

    Notes:
        We will use the xformers.ops.fmha.attn_bias.BlockDiagonalMask to perform
        cross attention

        To use it, we will construct the attn_bias using seq_lens.

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """

    # prepare for flash attention
    assert packed_value.size(-1) % 8 == 0, f"{packed_value.shape=}"
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(packed_value.dtype)
    packed_key = packed_key.to(packed_value.dtype)

    # get b_query info
    info_dict = packed_coord.get_localized_self_knn_info(
        k=k,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        attn_backend="xformers",
        debug=printout,
    )
    attn_biases = info_dict["attn_biases"]  # (num_chunks,)
    chunk_start_idxs = info_dict["chunk_start_idxs"]  # (num_chunks+1,)
    forward_idxs = info_dict["forward_idxs"]  # (bn,)
    backward_idxs = info_dict["backward_idxs"]  # (bn,)

    # no need to sort query, but need to sort kv
    packed_query = packed_query[forward_idxs]  # (bn, h, d)
    packed_key = packed_key[forward_idxs]  # (bn, h, d)
    packed_value = packed_value[forward_idxs]  # (bn, h, d)

    # run xformer attention
    # we do not need to sort, since packed point are sorted by bidx already
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(attn_biases)):
            _query = packed_query[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, m, h, d)
            _key = packed_key[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            _value = packed_value[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]].unsqueeze(
                0
            )  # (b=1, n, h, d)
            attn_bias = attn_biases[chunk_idx]
            # NOTE: we do not need the if branch to check whether _key will be zero-length as in self-attention,
            # there will always be at least one element in each query's cluster,
            # i.e., the key with the same coordinate as the query.
            out = xops.memory_efficient_attention(
                query=_query,  # (b=1, cm, h, d)
                key=_key,  # (b=1, n, h, d)
                value=_value,  # (b=1, n, h, d)
                attn_bias=attn_bias,
            )  # (b=1, cm, h, d)
            outs.append(out)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]  # (b=1, bm, h, d)
        out = out.squeeze(0)  # (bm, h, dv)

        # sort back
        out = out[backward_idxs]  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)
    return out


def localized_knn_self_softmax_attention_packed_flash_stacked(
    packed_coord: "PackedPoint",  # (n1+n2+...+nb, h, d)
    packed_qkv: torch.Tensor,  # (n1+n2+...+nb, 3qkv, h, d)
    k: T.Union[int, T.List[int]],
    use_cached: bool = False,
    cache_name: str = None,
    printout: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized knn self attention using flash attention.

    Args:
        packed_coord:
            (n1+n2+...nb, dn)
        packed_qkv:
            (bn, 3qkv, h, d)  qkv stacked
        k:
            int or list of (b,) number of clusters
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (n1+n2+...+nb, h, d) result of the self attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """
    # get b_query info
    info_dict = packed_coord.get_localized_self_knn_info(
        k=k,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        debug=printout,
        attn_backend="flash",
    )
    cu_seq_lens = info_dict["cu_seq_lens"]  # (num_chunks,), cumsum of seq_len in the chunk starts from 0
    max_seq_lens = info_dict["max_seq_lens"]  # (num_chunks,)
    chunk_start_idxs = info_dict["chunk_start_idxs"]  # (num_chunks+1,)
    forward_idxs = info_dict["forward_idxs"]  # (bn,)
    backward_idxs = info_dict["backward_idxs"]  # (bn,)

    # prepare for flash attention
    assert packed_qkv.size(-1) % 8 == 0
    ori_value_dtype = packed_qkv.dtype
    if packed_qkv.dtype not in (torch.float16, torch.bfloat16):
        packed_qkv = packed_qkv.to(torch.bfloat16)

    # sort
    packed_qkv = packed_qkv[forward_idxs]  # (bn, 3qkv, h, d)

    # run flash attention
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(cu_seq_lens)):
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                packed_qkv[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]],  # (bm', 3qkv, h, d)
                cu_seq_lens[chunk_idx],  # (num_cells + 1,)
                max_seq_lens[chunk_idx],  # int
            )  # (bm', h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)

        # sort back
        out = out[backward_idxs]  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)
    return out


def localized_knn_self_softmax_attention_packed_flash(
    packed_coord: "PackedPoint",  # (n1+n2+...+nb, h, d)
    packed_query: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_key: torch.Tensor,  # (n1+n2+...+nb, h, d)
    packed_value: torch.Tensor,  # (n1+n2+...+nb, h, d)
    k: T.Union[int, T.List[int]],
    use_cached: bool = False,
    cache_name: str = None,
    printout: bool = False,
) -> torch.Tensor:
    r"""
    Perform localized knn self attention using flash attention.

    Args:
        packed_coord:
            (n1+n2+...nb, dn)
        packed_query:
            (bn, h, d)
        packed_key:
            (bn, h, d)
        packed_value:
            (bn, h, d)
        k:
            int or list of (b,) number of clusters
        use_cached:
            if True, will use the saved `cached_name` in packed_kv_coord.
        cache_name:
            if given, will save the knn info into cache with the given name.

    Returns:
        (n1+n2+...+nb, h, d) result of the self attention

    Complexity:
        The computational complexity is O(sum_i m * ni), where m is the number of query, and
        ni is the number of points in bi.
    """
    # get b_query info
    info_dict = packed_coord.get_localized_self_knn_info(
        k=k,
        cache_name=cache_name,
        use_cached=use_cached,
        save_to_cache=cache_name is not None,
        debug=printout,
        attn_backend="flash",
    )
    cu_seq_lens = info_dict["cu_seq_lens"]  # (num_chunks,), cumsum of seq_len in the chunk starts from 0
    max_seq_lens = info_dict["max_seq_lens"]  # (num_chunks,)
    chunk_start_idxs = info_dict["chunk_start_idxs"]  # (num_chunks+1,)
    forward_idxs = info_dict["forward_idxs"]  # (bn,)
    backward_idxs = info_dict["backward_idxs"]  # (bn,)

    # prepare for flash attention
    assert packed_query.size(-1) % 8 == 0
    assert packed_key.size(-1) % 8 == 0
    assert packed_value.size(-1) % 8 == 0
    ori_value_dtype = packed_value.dtype
    if packed_value.dtype not in (torch.float16, torch.bfloat16):
        packed_value = packed_value.to(torch.bfloat16)
    packed_query = packed_query.to(dtype=packed_value.dtype)
    packed_key = packed_key.to(dtype=packed_value.dtype)

    # sort
    packed_query = packed_query[forward_idxs]  # (bn, h, d)
    packed_key = packed_key[forward_idxs]  # (bn, h, d)
    packed_value = packed_value[forward_idxs]  # (bn, h, d)

    # run flash attention
    with _nullcontext():
        outs = []
        for chunk_idx in range(len(cu_seq_lens)):
            out = flash_attn.flash_attn_varlen_func(
                q=packed_query[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]],  # (bm', h, d)
                k=packed_key[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]],
                v=packed_value[chunk_start_idxs[chunk_idx] : chunk_start_idxs[chunk_idx + 1]],
                cu_seqlens_q=cu_seq_lens[chunk_idx],  # (num_cells + 1,)
                cu_seqlens_k=cu_seq_lens[chunk_idx],  # (num_cells + 1,)
                max_seqlen_q=max_seq_lens[chunk_idx],  # int
                max_seqlen_k=max_seq_lens[chunk_idx],  # int
            )  # (bm', h, d)
            outs.append(out)
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # (bm, h, d)

        # sort back
        out = out[backward_idxs]  # (bm, h, dv)
        out = out.to(dtype=ori_value_dtype)
    return out


def knn_avg_downsampling(
    packed_coord: "PackedPoint",
    packed_feature: torch.Tensor,
    k: int,
    avg_coord: bool,
) -> T.Dict[str, T.Union[torch.Tensor, "PackedPoint"]]:
    r"""
    knn downsampling by averaging the coordinate and feature of points belonging to the same centroid.

    Procedure:
    - randomly select k points.
    - Each occupied voxel generates exactly one point by
      averaging all points inside

    Args:
        packed_coord:
            (n1+n2+...+nb, dn)
        packed_feature:
            (n1+n2+...+nb, *d)
        k:
            number of centroid (ie, number of points to outut)
        avg_coord:
            whether to average the coordinate or use the refernce points' coordinates
        save_to_cache:
            whether to save the index in the cache of packed_coord

    Returns:
        packed_coord:
            (num_occupied_cells, dn)
        packed_feature:
            (num_occupied_cells, *d)
    """

    # # randomly select k points
    # assert (packed_coord.seq_lens >= k).all()
    # with torch.no_grad():
    #     bidxs = packed_coord.batch_idxs  # (n1+n2+...+nb,)
    #     ridxs = torch.randperm(packed_coord.bn, device=packed_coord.device)  # (n1+n2+...+nb,)
    #     aidxs = torch.argsort(bidxs[ridxs], stable=True)   # (n1+n2+...+nb,)
    #     # ridxs[aidxs] is sorted by bidxs but shuffled within each sample
    #     # so we simple need to select the first k points from each sample

    # unpack to (b, max_n, *d)
    unpack_dict = packed_coord.unpack_arr(
        packed_arr=[packed_coord.coord, packed_feature],  # (num_occupied_cells, *d)
        max_size=None,
    )
    coord = unpack_dict["unpacked_arr"][0]  # (b, n, dn)
    feature = unpack_dict["unpacked_arr"][1]  # (b, n, *d)
    valid_mask = unpack_dict["valid_mask"]  # (b, n)
    b, n, dn = coord.shape

    assert valid_mask.all(), "currently only implement the simplest case"
    ridxs = torch.randperm(feature.shape[1])[:k]  # (k,)
    ref_coord = coord[:, ridxs]  # (b, k, dn)
    with _nullcontext():
        knn_out = pytorch3d.ops.knn_points(
            p1=coord.float(),  # (b, n, dn)
            p2=ref_coord.float(),  # (b, k, dn)
            K=1,
        )
        kidxs = knn_out.idx  # (b, n, 1)  the cluster index each point belong to

    if avg_coord:
        # average xyz to get new xyz (so implicitly weighted by sample density)
        new_coord = torch.zeros(b, k, dn, dtype=packed_coord.dtype, device=packed_coord.device)  # (b, k, dn)
        new_coord.scatter_reduce_(
            dim=0,
            index=kidxs.unsqueeze(-1).expand(b, n, dn),  # (b, n, dn)
            src=coord,  # (b, n, dn)
            reduce="mean",
            include_self=False,  # important, do not want to include 0 and the count
        )  # (b, k, dn)
    else:
        new_coord = ref_coord  # (b, k, dn)

    # average feature (so implicitly weighted by sample density)
    bn, *d_shape = packed_feature.shape
    assert bn == packed_coord.bn
    d = math.prod(d_shape)

    feat_mean = torch.zeros(b, k, d, dtype=packed_feature.dtype, device=packed_feature.device)  # (b, k, d)
    feat_mean.scatter_reduce_(
        dim=0,
        index=kidxs.unsqueeze(-1).expand(b, n, d),  # (b, n, d)
        src=feature.view(b, n, d),  # (b, n, d)
        reduce="mean",
        include_self=False,  # important, do not want to include 0 and the count
    )  # (b, k, d)
    # reshape d back to *d
    feat_mean = feat_mean.view(b, n, *d_shape)  # (b, n, *d)

    return dict(
        coord=new_coord,  # (b, k, dn)
        feature=feat_mean,  # (b, k, *d)
    )


def index_ranges(tensor, ranges, dim=0):
    """
    Index multiple ranges from a tensor along a specified dimension.

    Args:
        tensor:
            (n1 + n2 + ... + nd, ) Input tensor
        ranges:
            List of tuples (start, end) for each range
        dim:
            Dimension along which to index

    Returns:
        Concatenated tensor with all specified ranges
    """
    indices = torch.cat([torch.arange(start, end, device=tensor.device) for start, end in ranges])
    return torch.index_select(tensor, dim, indices)


def pixart_modulate(x, shift, scale, coord, debug_name=None):
    with _nullcontext():
        x = x.float()
        shift = shift.float()
        scale = scale.float()

        # Diagnostic: Check for extreme values
        x_abs_max = x.abs().max().item()
        scale_abs_max = scale.abs().max().item()
        shift_abs_max = shift.abs().max().item()
        has_bad = (
            x.isnan().any().item()
            or x.isinf().any().item()
            or scale.isnan().any().item()
            or scale.isinf().any().item()
            or shift.isnan().any().item()
            or shift.isinf().any().item()
        )

        if x_abs_max > 1000 or scale_abs_max > 10 or shift_abs_max > 1000 or has_bad:
            import warnings

            warnings.warn(
                f"[pixart_modulate{f' ({debug_name})' if debug_name else ''}] "
                f"x_max={x_abs_max:.2f}, scale_max={scale_abs_max:.2f}, shift_max={shift_abs_max:.2f}, has_nan_inf={has_bad}"
            )

        batched_x = coord.index_batches(torch.arange(coord.batch_size), x, unflatten=True)

        out_x = [bx * (1 + sc) + sh for bx, sc, sh in zip(batched_x, scale, shift)]

        result = torch.cat(out_x, dim=0)

        # Diagnostic: Check output
        result_abs_max = result.abs().max().item()
        result_has_nan = result.isnan().any().item()
        result_has_inf = result.isinf().any().item()

        if result_abs_max > 10000 or result_has_nan or result_has_inf:
            import warnings

            warnings.warn(
                f"[pixart_modulate{f' ({debug_name})' if debug_name else ''}] "
                f"OUTPUT: max={result_abs_max:.2f}, nan={result_has_nan}, inf={result_has_inf}"
            )

        return result


def packed_multiply(x, scale, coord, debug_name=None):
    with _nullcontext():
        x = x.float()
        scale = scale.float()

        # Diagnostic: Check for extreme values before multiplication
        x_abs_max = x.abs().max().item()
        scale_abs_max = scale.abs().max().item()
        x_has_nan = x.isnan().any().item()
        x_has_inf = x.isinf().any().item()
        scale_has_nan = scale.isnan().any().item()
        scale_has_inf = scale.isinf().any().item()

        if x_abs_max > 1000 or scale_abs_max > 10 or x_has_nan or x_has_inf or scale_has_nan or scale_has_inf:
            import warnings

            warnings.warn(
                f"[packed_multiply{f' ({debug_name})' if debug_name else ''}] "
                f"x: max={x_abs_max:.2f}, nan={x_has_nan}, inf={x_has_inf} | "
                f"scale: max={scale_abs_max:.2f}, nan={scale_has_nan}, inf={scale_has_inf}"
            )

        batched_x = coord.index_batches(torch.arange(coord.batch_size), x, unflatten=True)

        out_x = [bx * sc for bx, sc in zip(batched_x, scale)]

        result = torch.cat(out_x, dim=0)

        # Diagnostic: Check output
        result_abs_max = result.abs().max().item()
        result_has_nan = result.isnan().any().item()
        result_has_inf = result.isinf().any().item()

        if result_abs_max > 10000 or result_has_nan or result_has_inf:
            import warnings

            warnings.warn(
                f"[packed_multiply{f' ({debug_name})' if debug_name else ''}] "
                f"OUTPUT: max={result_abs_max:.2f}, nan={result_has_nan}, inf={result_has_inf}"
            )

        return result

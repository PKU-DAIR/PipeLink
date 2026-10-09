"""Device-side scheduler strategies named by partner contracts."""
from collections.abc import Mapping
from dataclasses import dataclass

import tilelang.language as T


_INPUTS = ("swizzle_size", "raster_order", "cluster_shape_m", "cluster_shape_n")


def _field(obj, name):
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def _select(condition, yes, no):
    return (yes if condition else no) if isinstance(condition, bool) else T.Select(condition, yes, no)


def _and(left, right):
    return (left and right) if isinstance(left, bool) and isinstance(right, bool) else T.And(left, right)


@dataclass(frozen=True)
class _ClcSm100:
    swizzle_size: object
    raster_order: object
    cluster_shape_m: object
    cluster_shape_n: object
    grid_m: object = None
    grid_n: object = None

    def _map(self, m, n, l, valid, offset_m=0, offset_n=0):
        """Pinned CUTLASS sm100_tile_scheduler.hpp:697–802, transpose:668–676.

        Use the executable mapping (788–789), not the comment's serpentine
        example: this revision does not implement a serpentine flip. Residual
        cluster rows/columns are unchanged, and AlongN transposes cluster
        coordinates while retaining the intra-cluster CTA remainder.
        """
        cm, cn, size = self.cluster_shape_m, self.cluster_shape_n, self.swizzle_size
        # Guard the divisor even in eagerly evaluated Select arms for size=0.
        divisor = _select(size > 0, size, 1)
        major, minor = m // cm, n // cn
        grid_m = T.get_block_extent(0) if self.grid_m is None else self.grid_m
        grid_n = T.get_block_extent(1) if self.grid_n is None else self.grid_n
        major_groups = (grid_m // cm) // divisor
        minor_groups = (grid_n // cn) // divisor
        swizzle = _and(size > 0, _and(major < major_groups * divisor, minor < minor_groups * divisor))
        new_major = _select(swizzle, major // divisor + major_groups * (minor % divisor), major)
        new_minor = _select(swizzle, major % divisor + (minor // divisor) * divisor, minor)
        m, n = new_major * cm + m % cm, new_minor * cn + n % cn
        along_n = self.raster_order == 1
        out_m = _select(along_n, (n // cn) * cm + m % cm, m)
        out_n = _select(along_n, (m // cm) * cn + n % cn, n)
        return valid, out_m + offset_m, out_n + offset_n, l

    def initial_tile(self, block_id_x, block_id_y, block_id_z):
        """CUTLASS initial_work_tile_info, sm100_tile_scheduler.hpp:367–370."""
        return self._map(block_id_x, block_id_y, block_id_z, True)

    def decode(self, clc_result_buffer_slice):
        """Decode first-CTA response and add this CTA's cluster offset (411–469)."""
        valid = T.clc_is_canceled(clc_result_buffer_slice) == 1
        m = T.clc_get_first_ctaid_x(clc_result_buffer_slice)
        n = T.clc_get_first_ctaid_y(clc_result_buffer_slice)
        l = T.clc_get_first_ctaid_z(clc_result_buffer_slice)
        rank = T.block_rank_in_cluster()
        return self._map(m, n, l, valid, rank % self.cluster_shape_m,
                         (rank // self.cluster_shape_m) % self.cluster_shape_n)


def clc_sm100_v1(partner_strategy, params):
    """Bind strategy inputs to Python constants or TileLang scalar expressions.

    ``raster_order`` is CUTLASS detail::RasterOrder (AlongM=0, AlongN=1),
    not RasterOrderOptions (which also contains Heuristic). The string names
    AlongM/AlongN are accepted for static specialization. Grid extents come
    from the active T.Kernel/T.ClusterKernel launch, never problem dimensions.
    """
    strategy_id = _field(partner_strategy, "id") if partner_strategy is not None else None
    if strategy_id != "clc-sm100-v1":
        raise NotImplementedError(f"unsupported scheduler strategy: {strategy_id!r}")
    if tuple(_field(partner_strategy, "inputs")) != _INPUTS:
        raise ValueError("clc-sm100-v1 strategy inputs differ from the pinned contract")
    values = {name: _field(params, name) for name in _INPUTS}
    # Extracted device roles execute in the linker's runtime grid, not their
    # authoring launch. Explicit runtime grid expressions override that launch.
    for name in ("grid_m", "grid_n"):
        values[name] = params.get(name) if isinstance(params, Mapping) else getattr(params, name, None)
    order = values["raster_order"]
    if isinstance(order, str):
        if order not in ("AlongM", "AlongN"):
            raise ValueError(f"unknown raster_order: {order!r}")
        values["raster_order"] = int(order == "AlongN")
    elif isinstance(order, int) and order not in (0, 1):
        raise ValueError("raster_order must be CUTLASS AlongM=0 or AlongN=1")
    for name in ("swizzle_size", "cluster_shape_m", "cluster_shape_n"):
        value = values[name]
        if isinstance(value, int) and value < (0 if name == "swizzle_size" else 1):
            raise ValueError(f"invalid {name}: {value}")
    return _ClcSm100(**values)

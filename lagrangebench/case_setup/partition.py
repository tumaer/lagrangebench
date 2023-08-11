from functools import partial
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import numpy as onp
from jax import jit
from jax_md import space
from jax_md.partition import (
    CellList,
    MaskFn,
    NeighborFn,
    NeighborList,
    NeighborListFns,
    NeighborListFormat,
    _displacement_or_metric_to_metric_sq,
    _neighboring_cells,
    _shift_array,
    cell_list,
    is_format_valid,
    is_sparse,
    neighbor_list,
)


def _scan_neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box_size: space.Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Dense,
    **static_kwargs,
) -> NeighborFn:
    """Modified JAX-MD neighbor list function that uses `lax.scan` to compute the
    distance between particles to save memory.

    Original: https://github.com/jax-md/jax-md/blob/main/jax_md/partition.py

    Returns a function that builds a list neighbors for collections of points.

    Neighbor lists must balance the need to be jit compatible with the fact that
    under a jit the maximum number of neighbors cannot change (owing to static
    shape requirements). To deal with this, our `neighbor_list` returns a
    `NeighborListFns` object that contains two functions: 1)
    `neighbor_fn.allocate` create a new neighbor list and 2) `neighbor_fn.update`
    updates an existing neighbor list. Neighbor lists themselves additionally
    have a convenience `update` member function.

    Note that allocation of a new neighbor list cannot be jit compiled since it
    uses the positions to infer the maximum number of neighbors (along with
    additional space specified by the `capacity_multiplier`). Updating the
    neighbor list can be jit compiled; if the neighbor list capacity is not
    sufficient to store all the neighbors, the `did_buffer_overflow` bit
    will be set to `True` and a new neighbor list will need to be reallocated.

    Here is a typical example of a simulation loop with neighbor lists:

    .. code-block:: python

        init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
        exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)

        nbrs = neighbor_fn.allocate(R)
        state = init_fn(random.PRNGKey(0), R, neighbor_idx=nbrs.idx)

        def body_fn(i, state):
        state, nbrs = state
        nbrs = nbrs.update(state.position)
        state = apply_fn(state, neighbor_idx=nbrs.idx)
        return state, nbrs

        step = 0
        for _ in range(20):
        new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
        if nbrs.did_buffer_overflow:
            nbrs = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += 1

    Args:
        displacement: A function `d(R_a, R_b)` that computes the displacement
        between pairs of points.
        box_size: Either a float specifying the size of the box or an array of
        shape `[spatial_dim]` specifying the box size in each spatial dimension.
        r_cutoff: A scalar specifying the neighborhood radius.
        dr_threshold: A scalar specifying the maximum distance particles can move
        before rebuilding the neighbor list.
        capacity_multiplier: A floating point scalar specifying the fractional
        increase in maximum neighborhood occupancy we allocate compared with the
        maximum in the example positions.
        disable_cell_list: An optional boolean. If set to `True` then the neighbor
        list is constructed using only distances. This can be useful for
        debugging but should generally be left as `False`.
        mask_self: An optional boolean. Determines whether points can consider
        themselves to be their own neighbors.
        custom_mask_function: An optional function. Takes the neighbor array
        and masks selected elements. Note: The input array to the function is
        `(n_particles, m)` where the index of particle 1 is in index in the first
        dimension of the array, the index of particle 2 is given by the value in
        the array
        fractional_coordinates: An optional boolean. Specifies whether positions
        will be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`.
        If this is set to True then the `box_size` will be set to `1.0` and the
        cell size used in the cell list will be set to `cutoff / box_size`.
        format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
        for details about the different choices for formats. Defaults to `Dense`.
        **static_kwargs: kwargs that get threaded through the calculation of
        example positions.
    Returns:
        A NeighborListFns object that contains a method to allocate a new neighbor
        list and a method to update an existing neighbor list.
    """
    is_format_valid(format)
    box_size = lax.stop_gradient(box_size)
    r_cutoff = lax.stop_gradient(r_cutoff)
    dr_threshold = lax.stop_gradient(dr_threshold)

    box_size = jnp.float32(box_size)

    cutoff = r_cutoff + dr_threshold
    cutoff_sq = cutoff**2
    threshold_sq = (dr_threshold / jnp.float32(2)) ** 2
    metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

    cell_size = cutoff
    if fractional_coordinates:
        cell_size = cutoff / box_size
        box_size = (
            jnp.float32(box_size)
            if onp.isscalar(box_size)
            else onp.ones_like(box_size, jnp.float32)
        )

    use_cell_list = jnp.all(cell_size < box_size / 3.0) and not disable_cell_list
    if use_cell_list:
        cl_fn = cell_list(box_size, cell_size, capacity_multiplier)

    @jit
    def candidate_fn(position: jnp.ndarray) -> jnp.ndarray:
        candidates = jnp.arange(position.shape[0])
        return jnp.broadcast_to(
            candidates[None, :], (position.shape[0], position.shape[0])
        )

    @jit
    def cell_list_candidate_fn(cl: CellList, position: jnp.ndarray) -> jnp.ndarray:
        N, dim = position.shape

        idx = cl.id_buffer

        cell_idx = [idx]

        for dindex in _neighboring_cells(dim):
            if onp.all(dindex == 0):
                continue
        cell_idx += [_shift_array(idx, dindex)]

        cell_idx = jnp.concatenate(cell_idx, axis=-2)
        cell_idx = cell_idx[..., jnp.newaxis, :, :]
        cell_idx = jnp.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])

        def copy_values_from_cell(value, cell_value, cell_id):
            scatter_indices = jnp.reshape(cell_id, (-1,))
            cell_value = jnp.reshape(cell_value, (-1,) + cell_value.shape[-2:])
            return value.at[scatter_indices].set(cell_value)

        neighbor_idx = jnp.zeros((N + 1,) + cell_idx.shape[-2:], jnp.int32)
        neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
        return neighbor_idx[:-1, :, 0]

    @jit
    def mask_self_fn(idx: jnp.ndarray) -> jnp.ndarray:
        self_mask = idx == jnp.reshape(
            jnp.arange(idx.shape[0], dtype=jnp.int32), (idx.shape[0], 1)
        )
        return jnp.where(self_mask, idx.shape[0], idx)

    @jit
    def prune_neighbor_list_dense(
        position: jnp.ndarray, idx: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        d = partial(metric_sq, **kwargs)
        d = space.map_neighbor(d)

        N = position.shape[0]
        neigh_position = position[idx]
        dR = d(position, neigh_position)

        mask = (dR < cutoff_sq) & (idx < N)
        out_idx = N * jnp.ones(idx.shape, jnp.int32)

        cumsum = jnp.cumsum(mask, axis=1)
        index = jnp.where(mask, cumsum - 1, idx.shape[1] - 1)
        p_index = jnp.arange(idx.shape[0])[:, None]
        out_idx = out_idx.at[p_index, index].set(idx)
        max_occupancy = jnp.max(cumsum[:, -1])

        return out_idx, max_occupancy

    @jit
    def prune_neighbor_list_sparse(
        position: jnp.ndarray, idx: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        d = partial(metric_sq, **kwargs)
        d = space.map_bond(d)

        N = position.shape[0]
        sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape)

        sender_idx = jnp.reshape(sender_idx, (-1,))
        receiver_idx = jnp.reshape(idx, (-1,))
        dR = d(position[sender_idx], position[receiver_idx])

        mask = (dR < cutoff_sq) & (receiver_idx < N)
        if format is NeighborListFormat.OrderedSparse:
            mask = mask & (receiver_idx < sender_idx)

        out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)

        cumsum = jnp.cumsum(mask)
        index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)
        receiver_idx = out_idx.at[index].set(receiver_idx)
        sender_idx = out_idx.at[index].set(sender_idx)
        max_occupancy = cumsum[-1]

        return jnp.stack((receiver_idx, sender_idx)), max_occupancy

    def neighbor_list_fn(
        position: jnp.ndarray,
        neighbors: Optional[NeighborList] = None,
        extra_capacity: int = 0,
        **kwargs,
    ) -> NeighborList:
        nbrs = neighbors

        def neighbor_fn(position_and_overflow, max_occupancy=None):
            position, overflow = position_and_overflow
            N = position.shape[0]

            if use_cell_list:
                if neighbors is None:
                    cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
                else:
                    cl = cl_fn.update(position, neighbors.cell_list_capacity)
                overflow = overflow | cl.did_buffer_overflow
                idx = cell_list_candidate_fn(cl, position)
                cl_capacity = cl.cell_capacity
            else:
                cl_capacity = None
                idx = candidate_fn(position)

            if mask_self:
                idx = mask_self_fn(idx)
            if custom_mask_function is not None:
                idx = custom_mask_function(idx)

            if is_sparse(format):
                idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)
            else:
                idx, occupancy = prune_neighbor_list_dense(position, idx, **kwargs)

            if max_occupancy is None:
                _extra_capacity = (
                    extra_capacity if not is_sparse(format) else N * extra_capacity
                )
                max_occupancy = int(occupancy * capacity_multiplier + _extra_capacity)
                if max_occupancy > idx.shape[-1]:
                    max_occupancy = idx.shape[-1]
                if not is_sparse(format):
                    capacity_limit = N - 1 if mask_self else N
                elif format is NeighborListFormat.Sparse:
                    capacity_limit = N * (N - 1) if mask_self else N ** 2
                else:
                    capacity_limit = N * (N - 1) // 2
                if max_occupancy > capacity_limit:
                    max_occupancy = capacity_limit
            idx = idx[:, :max_occupancy]
            update_fn = neighbor_list_fn if neighbors is None else neighbors.update_fn
            return NeighborList(
                idx,
                position,
                overflow | (occupancy > max_occupancy),
                cl_capacity,
                max_occupancy,
                format,
                update_fn,
            )  # pytype: disable=wrong-arg-count

        if nbrs is None:
            return neighbor_fn((position, False))

        neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

        d = partial(metric_sq, **kwargs)

        # replace vmap with scan to save memory
        def distance_fn(x, y):
            return lax.scan(lambda _, x: (None, d(*x)), None, (x, y))[1]

        return lax.cond(
            jnp.any(distance_fn(position, nbrs.reference_position) > threshold_sq),
            (position, nbrs.did_buffer_overflow),
            neighbor_fn,
            nbrs,
            lambda x: x,
        )

    def allocate_fn(
        position: jnp.ndarray, extra_capacity: int = 0, **kwargs
    ) -> NeighborList:
        return neighbor_list_fn(position, extra_capacity=extra_capacity, **kwargs)

    def update_fn(
        position: jnp.ndarray, neighbors: NeighborList, **kwargs
    ) -> NeighborList:
        return neighbor_list_fn(position, neighbors, **kwargs)

    return NeighborListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


def _matscipy_neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box_size: space.Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Dense,
    **static_kwargs,
) -> NeighborFn:
    pbc = static_kwargs["pbc"]
    num_particles_max = static_kwargs["num_particles_max"]

    from matscipy.neighbours import neighbour_list as matscipy_nl

    assert box_size.ndim == 1 and (len(box_size) in [2, 3])
    if len(box_size) == 2:
        box_size = np.pad(box_size, (0, 1), mode="constant", constant_values=1.0)
        box_size = np.eye(3) * box_size
    if len(pbc) == 2:
        pbc = np.pad(pbc, (0, 1), mode="constant", constant_values=False)

    def matscipy_wrapper(positions, idx):
        res = matscipy_nl(
            "ij", cutoff=r_cutoff, positions=positions, cell=box_size, pbc=True
        )

        # buffer overflow
        buffer_overflow = (
            jnp.array(True) if res.shape[1] > idx.shape[1] else jnp.array(False)
        )

        # edge list
        idx_new = jnp.ones_like(idx) * num_particles_max
        idx_new = idx_new.at[:, : res.shape[0]].set(res)

        return idx_new, buffer_overflow

    @jax.jit
    def update_fn(
        position: jnp.ndarray, neighbors: NeighborList, **kwargs
    ) -> NeighborList:
        if position.shape[1] == 2:
            position = np.pad(
                position, ((0, 0), (0, 1)), mode="constant", constant_values=0.5
            )

        shape_edgelist = jax.ShapeDtypeStruct(
            neighbors.idx.shape, dtype=neighbors.idx.dtype
        )
        shape_overflow = jax.ShapeDtypeStruct((), dtype=bool)
        shape = (shape_edgelist, shape_overflow)
        idx, buffer_overflow = jax.pure_callback(
            matscipy_wrapper, shape, position, neighbors.idx
        )

        return NeighborList(
            idx,
            position,
            buffer_overflow,
            None,
            None,
            None,
            update_fn,
        )

    def allocate_fn(
        position: jnp.ndarray, extra_capacity: int = 0, **kwargs
    ) -> NeighborList:
        num_particles = int(position[-1, 0])
        position = position[:num_particles]

        if position.shape[1] == 2:
            position = np.pad(
                position, ((0, 0), (0, 1)), mode="constant", constant_values=0.5
            )

        edge_list = matscipy_nl(
            "ij", cutoff=r_cutoff, positions=position, cell=box_size, pbc=pbc
        )
        edge_list = np.asarray(edge_list, dtype=np.int32)
        # in case this is a (2,M) pair list, we pad with N and capacity_multiplier
        res = num_particles * jnp.ones(
            (2, round(edge_list.shape[1] * capacity_multiplier + extra_capacity)),
            np.int32,
        )
        res = res.at[:, : edge_list.shape[1]].set(edge_list)
        return NeighborList(
            res,
            position,
            jnp.array(False),
            None,
            None,
            None,
            update_fn,
        )

    return NeighborListFns(allocate_fn, update_fn)


BACKENDS = {
    "jaxmd_vmap": neighbor_list,
    "jaxmd_scan": _scan_neighbor_list,
    "matscipy": _matscipy_neighbor_list,
}


def neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box_size: space.Box,
    r_cutoff: float,
    backend: str = "jaxmd_vmap",
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Dense,
    num_particles_max: int = None,
    pbc: jnp.ndarray = None,
) -> NeighborFn:
    """Neighbor lists wrapper.

    Args:
        backend: The backend to use. One of "jaxmd_vmap", "jaxmd_scan", "matscipy".
            * "jaxmd_vmap": Default jax-md neighbor list. Uses vmap. Fast.
            * "jaxmd_scan": Modified jax-md neighbor list. Uses scan. Memory efficient.
            * "matscipy": Matscipy neighbor list. Runs on cpu, allows dynamic shapes.
    """
    assert backend in BACKENDS, f"Unknown backend {backend}"

    return BACKENDS[backend](
        displacement_or_metric,
        box_size,
        r_cutoff,
        dr_threshold,
        capacity_multiplier,
        disable_cell_list,
        mask_self,
        custom_mask_function,
        fractional_coordinates,
        format,
        num_particles_max=num_particles_max,
        pbc=pbc,
    )

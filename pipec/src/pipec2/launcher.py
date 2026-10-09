"""Launch linked kernels through the shared CUDA backend, with or without host ABI v3."""
from __future__ import annotations

import json
from contextlib import ExitStack
import math
import operator
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import DTypeLike

from .host import abi as A
from .model import ClauseFailure, LegalitySet, LinkError, LinkResult, ParamsLayout, register_clause

register_clause("L-LAUNCH", phase="launch", description="Invalid launch request or launcher failure")


def _operation_error(message: str) -> LinkError:
    return _failure("L-LAUNCH", "launch", message)


def _failure(clause_id: str, path: str, message: str, **detail: Any) -> LinkError:
    return LinkError([ClauseFailure(clause_id, path, message, detail)])


def _load_link_result(path: Path) -> LinkResult:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _operation_error(f"cannot read link manifest {path}: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
        result = LinkResult.from_json_obj(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _operation_error(f"invalid link manifest {path}: {exc}") from exc
    return result


def _dimension3(value: Sequence[int], path: str) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise _operation_error(f"{path} must contain exactly three dimensions")
    dimensions = tuple(value)
    if any(type(item) is not int or item <= 0 for item in dimensions):
        raise _operation_error(f"{path} dimensions must be positive integers")
    return dimensions  # type: ignore[return-value]


def _select_cluster(
    declared: tuple[int, int, int] | LegalitySet,
    requested: Sequence[int] | None,
) -> tuple[int, int, int]:
    if isinstance(declared, LegalitySet):
        cluster = (
            (declared.x_multiple_of, 1, 1)
            if requested is None
            else _dimension3(requested, "cluster")
        )
        if cluster[0] % declared.x_multiple_of != 0 or any(
            actual > maximum
            for actual, maximum in zip(cluster, declared.max, strict=True)
        ):
            raise _failure(
                "L-TOPO", "launch.cluster", "requested cluster is outside the linked legality set",
                requested=cluster, x_multiple_of=declared.x_multiple_of, max=declared.max,
            )
        return cluster
    fixed = _dimension3(declared, "link.cluster")
    if requested is not None and _dimension3(requested, "cluster") != fixed:
        raise _failure(
            "L-TOPO", "launch.cluster", "requested cluster differs from the fixed linked cluster",
            requested=tuple(requested), fixed=fixed,
        )
    return fixed


def _assemble_hostless_params(layout: ParamsLayout, blobs: Mapping[str, bytes]) -> A.aligned_buffer:
    from .host.broker import _write_stage_blobs

    entries = {name: (offset, size, align) for name, offset, size, align in layout.entries}
    required = {name for name, (_, size, _) in entries.items() if size}
    missing, unknown = required - blobs.keys(), blobs.keys() - entries.keys()
    if missing or unknown:
        raise _failure("L-PARAMS", "launch.params", "stage blobs do not cover the linked params",
                       missing=sorted(missing), unknown=sorted(unknown))
    try:
        image = A.aligned_buffer(layout.total_bytes, layout.align)
    except ValueError as exc:
        raise _failure("L-PARAMS", "launch.params", str(exc)) from exc
    for name in blobs:
        offset, size, align = entries[name]
        if (align <= 0 or align & (align - 1) or align > layout.align or
                offset < 0 or offset % align or offset + size > image.size):
            raise _failure("L-PARAMS", name, "blob region exceeds or misaligns aggregate params",
                           offset=offset, bytes=size, align=align)
    _write_stage_blobs(image, layout.entries, blobs,
                       lambda name, message, **detail: _failure("L-PARAMS", name, message, **detail))
    return image


class HostlessSession:
    """Pinned buffers, opaque params, and geometry on one owned CUDA stream.

    Open uploads inputs and zeroes allocation specs once; repeated launches retain
    buffer contents. Readback synchronizes and returns an independent NumPy array.
    """

    def __init__(self, backend: Any, lifetime: ExitStack, addresses: dict[str, int],
                 buffers: dict[str, tuple[tuple[int, ...], np.dtype, int]],
                 params: A.aligned_buffer, geometry: A.plk_geometry_v2, *,
                 kernel_symbol: str, device: int, stream: int):
        self.addresses = addresses
        self._backend = backend
        self._lifetime = lifetime
        self._buffers = buffers
        self._params = params
        self._geometry = geometry
        self._kernel_symbol = kernel_symbol
        self._device = device
        self._stream = stream

    def with_image(self, cubin: str | Path) -> HostlessSession:
        """Load another kernel image sharing this invocation's buffers, params and stream.

        The owning session must outlive every image view. Closing a view unloads
        only its backend; allocations remain owned by the original session.
        """
        from .host.cuda import CudaBackend

        image = Path(cubin)
        if not image.is_file():
            raise _operation_error(f"linked cubin does not exist: {image}")
        geometry = self._geometry
        with ExitStack() as lifetime:
            backend = lifetime.enter_context(CudaBackend(
                image, self._kernel_symbol, device=self._device,
                dynamic_smem_bytes=geometry.dynamic_smem_bytes,
                cluster=(geometry.cluster.x, geometry.cluster.y, geometry.cluster.z),
                block=(geometry.block.x, geometry.block.y, geometry.block.z),
            ))
            return HostlessSession(
                backend, lifetime.pop_all(), self.addresses, self._buffers, self._params, geometry,
                kernel_symbol=self._kernel_symbol, device=self._device, stream=self._stream,
            )

    def launch(self) -> None:
        self._backend.launch(self._params, self._geometry, self._stream)

    def synchronize(self) -> None:
        self._backend.synchronize()

    def read(self, name: str) -> np.ndarray:
        shape, dtype, size = self._buffers[name]
        self.synchronize()
        data = self._backend.readback(self.addresses[name], size)
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    def write(self, name: str, value: np.ndarray) -> None:
        """Overwrite one buffer (e.g. poison an output between images sharing it)."""
        shape, dtype, size = self._buffers[name]
        data = np.ascontiguousarray(value, dtype=dtype).reshape(shape).tobytes()
        self.synchronize()
        self._backend.upload(self.addresses[name], data)

    def time(self, repetitions: int) -> float:
        """Return total milliseconds for a batch bracketed by one CUDA event pair."""
        if type(repetitions) is not int or repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")
        backend = self._backend
        with ExitStack() as events:
            start = backend.event_create()
            events.callback(backend.event_destroy, start)
            end = backend.event_create()
            events.callback(backend.event_destroy, end)
            backend.event_record(start, self._stream)
            for _ in range(repetitions):
                self.launch()
            backend.event_record(end, self._stream)
            backend.event_synchronize(end)
            elapsed = backend.event_elapsed_time(start, end)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise _operation_error(f"CUDA event batch duration must be finite and positive, got {elapsed}")
            return elapsed

    def close(self) -> None:
        self._lifetime.close()

    def __enter__(self) -> HostlessSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._lifetime.__exit__(*exc)


def open_hostless(
    link_json_path: str | Path, *,
    buffers: Mapping[str, np.ndarray | tuple[int | Sequence[int], DTypeLike]],
    params: Callable[[Mapping[str, int]], Mapping[str, bytes]],
    grid: tuple[int, int, int],
    cluster: Sequence[int] | None = None,
    cubin: Path | None = None,
    device: int = 0,
) -> HostlessSession:
    """Allocate every buffer, bind stage blobs to their addresses, and open a session."""
    from .host.cuda import CudaBackend

    link_path = Path(link_json_path).resolve()
    result = _load_link_result(link_path)
    if result.host is not None:
        raise _operation_error("open_hostless refuses host ABI links; use launch or open_host")
    grid = _dimension3(grid, "grid")
    cluster = _select_cluster(result.cluster, cluster)
    image = link_path.with_name("pipeline.cubin") if cubin is None else Path(cubin)
    if not image.is_file():
        raise _operation_error(f"linked cubin does not exist: {image}")
    block = (result.threads, 1, 1)
    geometry = A.plk_geometry_v2(
        geometry_present=1, launch_required=1, grid=A.plk_dim3_v2(*grid),
        block=A.plk_dim3_v2(*block), cluster=A.plk_dim3_v2(*cluster),
        dynamic_smem_bytes=result.dyn_smem_bytes,
    )
    with ExitStack() as lifetime:
        backend = lifetime.enter_context(CudaBackend(
            image, result.kernel_symbol, device=device, dynamic_smem_bytes=result.dyn_smem_bytes,
            cluster=cluster, block=block,
        ))
        addresses: dict[str, int] = {}
        metadata: dict[str, tuple[tuple[int, ...], np.dtype, int]] = {}
        for name, value in buffers.items():
            if isinstance(value, np.ndarray):
                shape, dtype = value.shape, value.dtype
            else:
                shape_spec, dtype_spec = value
                shape = ((operator.index(shape_spec),) if isinstance(shape_spec, (int, np.integer))
                         else tuple(operator.index(extent) for extent in shape_spec))
                dtype = np.dtype(dtype_spec)
            if any(extent < 0 for extent in shape) or dtype.hasobject or not dtype.itemsize:
                raise _operation_error(f"buffer {name!r} requires nonnegative extents and a fixed non-object dtype")
            size = math.prod(shape) * dtype.itemsize
            addresses[name] = backend.allocate(size, max(1, dtype.alignment))
            metadata[name] = (shape, dtype, size)
            if isinstance(value, np.ndarray):
                backend.upload(addresses[name], value.tobytes(order="C"))
            else:
                backend.zero(addresses[name], size, backend.stream_token)
        # Params builders may encode driver objects (tensor maps) against the session's context.
        with backend.current():
            built = params(addresses)
        image_params = _assemble_hostless_params(result.params, built)
        return HostlessSession(
            backend, lifetime.pop_all(), addresses, metadata, image_params, geometry,
            kernel_symbol=result.kernel_symbol, device=device, stream=backend.stream_token,
        )


def run_hostless(
    link_json_path: str | Path, *,
    buffers: Mapping[str, np.ndarray | tuple[int | Sequence[int], DTypeLike]],
    params: Callable[[Mapping[str, int]], Mapping[str, bytes]],
    grid: tuple[int, int, int],
    cluster: Sequence[int] | None = None,
    cubin: Path | None = None,
    device: int = 0,
    outputs: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Open, launch once, read selected buffers (all by default), and close."""
    names = list(buffers if outputs is None else outputs)
    unknown = set(names) - buffers.keys()
    if unknown:
        raise _operation_error(f"unknown output buffers: {sorted(unknown)}")
    with open_hostless(link_json_path, buffers=buffers, params=params, grid=grid,
                       cluster=cluster, cubin=cubin, device=device) as session:
        session.launch()
        session.synchronize()
        return {name: session.read(name) for name in names}


def launch(
    link_json_path: str | Path,
    blob_paths: Mapping[str, Path] | None = None,
    cluster: Sequence[int] | None = None,
    *,
    provider_paths: Mapping[str, Path] | None = None,
    provider_arg_paths: Mapping[str, Path] | None = None,
    host_resources: Mapping[str, Any] | None = None,
    host_domains: Mapping[str, Any] | None = None,
    host_scalars: Mapping[str, Any] | None = None,
    host_inputs_path: str | Path | None = None,
    device: int = 0,
    buffers: Mapping[str, np.ndarray | tuple[int | Sequence[int], DTypeLike]] | None = None,
    params: Callable[[Mapping[str, int]], Mapping[str, bytes]] | None = None,
    grid: tuple[int, int, int] | None = None,
    cubin: Path | None = None,
    output_paths: Mapping[str, Path] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Launch once through the host broker or the host-less buffer/params route."""
    link_path = Path(link_json_path).resolve()
    result = _load_link_result(link_path)
    if result.host is None:
        if provider_paths or provider_arg_paths or host_resources or host_domains or host_scalars or host_inputs_path is not None:
            raise _operation_error("host-less links do not accept host providers, resources, or scalars")
        if grid is None:
            raise _operation_error("host-less launch requires grid")
        if params is not None and blob_paths:
            raise _operation_error("params builder and blob paths are mutually exclusive")
        if params is None:
            params = lambda addresses: {name: Path(path).read_bytes()
                                        for name, path in (blob_paths or {}).items()}
        selected_cluster = _select_cluster(result.cluster, cluster)
        outputs = run_hostless(link_path, buffers=buffers or {}, params=params, grid=grid,
                               cluster=selected_cluster, cubin=cubin, device=device,
                               outputs=output_paths or {})
        for name, path in (output_paths or {}).items():
            with Path(path).open("wb") as stream:
                np.save(stream, outputs[name], allow_pickle=False)
        report = {"geometry": {"grid": grid, "block": (result.threads, 1, 1),
                               "cluster": selected_cluster, "dynamic_smem_bytes": result.dyn_smem_bytes},
                  "outputs": {name: str(path) for name, path in (output_paths or {}).items()}}
        return subprocess.CompletedProcess(["pipec2-hostless", str(link_path)], 0,
                                           json.dumps(report, sort_keys=True) + "\n", "")
    if buffers or params is not None or grid is not None or cubin is not None or output_paths:
        raise _operation_error("host ABI links use providers and host inputs, not host-less launch options")
    return _launch_host(
        link_path, result, _select_cluster(result.cluster, cluster),
        provider_paths or {}, provider_arg_paths or {}, blob_paths or {}, host_resources or {},
        host_domains or {}, host_scalars or {}, host_inputs_path, device,
    )


def _launch_host(
    link_path: Path, result: LinkResult, cluster: tuple[int, int, int],
    providers: Mapping[str, Path], arg_paths: Mapping[str, Path], blob_paths: Mapping[str, Path],
    resources: Mapping[str, Any], domains: Mapping[str, Any], scalars: Mapping[str, Any],
    inputs_path: str | Path | None, device: int,
) -> subprocess.CompletedProcess[str]:
    from .host.broker import open_host
    from .host.cuda import CudaBackend
    from .host.fixture import Fixture, load_inputs

    cubin = link_path.with_name("pipeline.cubin")
    if not cubin.is_file():
        raise _operation_error(f"linked cubin does not exist: {cubin}")
    try:
        args = {name: Path(path).read_bytes() for name, path in arg_paths.items()}
        blobs = {name: Path(path).read_bytes() for name, path in blob_paths.items()}
    except OSError as exc:
        raise _operation_error(f"cannot read owner arguments or stage blobs: {exc}") from exc
    output_paths: Mapping[str, Path] = {}
    with CudaBackend(cubin, result.kernel_symbol, device=device,
                     dynamic_smem_bytes=result.dyn_smem_bytes, cluster=cluster,
                     block=(result.threads, 1, 1)) as backend:
        with Fixture(backend) as fixture:
            if inputs_path is not None:
                if resources or scalars:
                    raise _failure("L-HOSTCHAN", "launch", "host input file cannot override caller inputs",
                                   subreason="inventory")
                scalars, output_paths = load_inputs(Path(inputs_path), fixture)
                resources = fixture.resources
            with open_host(result, providers, args, resources, scalars, backend, cluster, blobs=blobs,
                           domains=domains) as session:
                session.launch(backend.stream_token)
                backend.synchronize()
                outputs = fixture.readback()
                report = {
                    "geometry": session.geometry,
                    "toolchain_provenance": session.toolchain_provenance,
                    "transfer_reports": session.transfer_reports,
                    "readback_bytes": {name: data.hex() for name, data in outputs.items()},
                }
    for name, output_path in output_paths.items():
        try:
            output_path.write_bytes(outputs[name])
        except OSError as exc:
            raise _operation_error(f"cannot write host output {name}: {exc}") from exc
    return subprocess.CompletedProcess(["pipec2-host", str(link_path)], 0,
                                       json.dumps(report, sort_keys=True) + "\n", "")


__all__ = ["HostlessSession", "launch", "open_hostless", "run_hostless"]

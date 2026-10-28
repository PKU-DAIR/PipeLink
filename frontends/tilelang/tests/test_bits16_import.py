"""Packed f16/bf16 imports retain floating fragments and transport uint16 bits."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import tilelang.language as T
from tilelang import tvm

import pipelink_tilelang as pl
from pipelink_tilelang.capture import _nodes
from pipelink_tilelang.explicit import emit
from pipec2.schema import load_stage_manifest
from pipec2.tensor.lower import _declaration
from test_contract_tensor_import import ROOT, ROW_REGISTERS, _lower, _partner


def _consumer(partner, dtype):
    channel = pl.import_tensor(partner, "tile")

    def program():
        @T.prim_func
        def consume(output: T.Tensor((128, 2), dtype), tag: T.int32):
            with T.Kernel(1, threads=128):
                fragment = channel.fragment(specialize=True)
                with channel.imported(fragment):
                    T.copy(fragment, output)
        return consume

    stage = pl.contract_stage("test.bits16-consumer", "1.0.0", program)
    return SimpleNamespace(STAGE=stage, program=stage.program, channel=channel)


def _import_destination(module, generated):
    function = next(iter(generated[None]["modules"]["initial"].functions.values()))
    call, = [node for node in _nodes(function.body, tvm.tirx.Call)
             if getattr(node.op, "name", None) == "tirx.call_extern"
             and node.args[0].value == module.channel.symbol]
    return call.args[2]


@pytest.mark.parametrize("dtype,tilelang_dtype", [("bf16", "bfloat16"), ("f16", "float16")])
@pytest.mark.parametrize("storage", ["reg", "smem"])
def test_packed_float16_import_uses_bits_container(tmp_path, monkeypatch, dtype, tilelang_dtype, storage):
    from tilelang.env import TILELANG_TEMPLATE_PATH
    monkeypatch.syspath_prepend(str(ROOT))
    from tools.pipelink_env import load
    monkeypatch.setenv("PIPELINK_CUDA_HOME", load(ROOT)["PIPELINK_CUDA_HOME"])
    monkeypatch.setattr(pl, "context", pl.context)
    pl.set_context("sm_90a", None)

    def pack(source, destination, *flags):
        result = subprocess.run(
            [str(Path(sys.executable).with_name("pipec2")), "pack", str(source),
             "--profile", "sm_90a", "-o", str(destination), *flags],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    flags = []
    if dtype == "f16":
        flags.extend(["-D", "EXPORT_F16=1"])
    if storage == "smem":
        flags.extend(["-D", "EXPORT_SMEM=1"])
    producer = tmp_path / "producer.plink"
    pack(Path(__file__).parent / "fixtures/bits16_producer.cu", producer, *flags)
    partner = pl.Partner.load(producer)
    assert partner.manifest.tensor.exports[0].identity.dtype == dtype
    assert partner.manifest.tensor.exports[0].placements[0].space == storage

    module = _consumer(partner, tilelang_dtype)
    generated = _lower(module, "sm_90a")
    destination = _import_destination(module, generated)
    assert isinstance(destination, tvm.tirx.Call)
    assert destination.args[0].value == "pipelink_tilelang_bits16"
    assert destination.args[1].same_as(module.channel._fragment.data)
    assert str(module.channel._fragment.dtype) == tilelang_dtype

    projection = emit(tmp_path, module, generated)[module.STAGE.stage_id]
    consumer = tmp_path / "consumer.plink"
    pack(projection["source"], consumer, "-I", TILELANG_TEMPLATE_PATH,
         "-I", str(ROOT / "third_party/cutlass/include"))
    with zipfile.ZipFile(consumer) as archive:
        manifest = load_stage_manifest(json.loads(archive.read("manifest.json")))
    imported, = manifest.tensor.imports
    assert imported.identity.dtype == dtype
    destination_param, = [param for param in imported.callable.params if param.meaning == "dest_fragment"]
    assert destination_param.ctype == "uint16_t*"
    _declaration(imported)


def test_float32_destination_stays_unwrapped(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "context", pl.context)
    pl.set_context("sm_90a", None)
    partner = _partner(tmp_path, "float32", "tile", [("m", 128), ("n", 2)], "float32_args",
                       [("int32_t", "tag")], profile="sm_90a", placement=ROW_REGISTERS)
    module = _consumer(partner, "float32")
    generated = _lower(module, "sm_90a")
    assert _import_destination(module, generated).same_as(module.channel._fragment.data)

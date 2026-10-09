from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import ctypes as C
import json
import struct

import numpy as np
import pytest

from pipec2 import cli, launcher
from pipec2.host import cuda
from pipec2.host.contracts import host_params_layout
from pipec2.host.profile import DTYPES
from pipec2.model import LinkError, LinkResult, ParamsLayout
from test_host_broker import MemoryBackend, StubResidue, owner_args, trio


def _result(host, params):
    return LinkResult('pipelink_kernel',32,32,0,(1,1,1),params,{},{},{},[],(0,0,0),{},{},None,host)


def test_open_hostless_refuses_host_link(tmp_path):
    path = tmp_path / 'link.json'
    path.write_text(json.dumps(_result({}, ParamsLayout(0,1,[])).to_json_obj()))
    with pytest.raises(LinkError) as caught:
        launcher.open_hostless(path, buffers={}, params=lambda addresses: {}, grid=(1,1,1))
    assert caught.value.failures[0].clause_id == 'L-LAUNCH'


@pytest.fixture
def launch_case(trio,tmp_path,monkeypatch):
    baseline, providers, _ = trio
    host = deepcopy(baseline.host)
    blob_offset = (host['params']['bytes']+63) & -64
    total = blob_offset + 64
    host['params'] = {'bytes':total, 'align':64, 'blobs':[
        {'stage_id':'stub.instruction', 'offset':blob_offset, 'bytes':8, 'align':8},
        {'stage_id':'stub.zero', 'offset':total, 'bytes':0, 'align':8},
    ]}
    result = _result(host,host_params_layout(host))
    path = tmp_path / 'link.json'
    path.write_text(json.dumps(result.to_json_obj()))
    (tmp_path/'pipeline.cubin').write_bytes(b'CPU backend consumes the published params image')
    instruction = tmp_path / 'instruction.bin'
    instruction.write_bytes(struct.pack('<Q',5))
    arg_paths = {}
    for name, data in owner_args().items():
        arg_paths[name] = tmp_path / (name+'.args')
        arg_paths[name].write_bytes(data)
    (tmp_path/'input.bin').write_bytes(bytes(64))
    resource = {'data':'input.bin', 'allocation_bytes':64, 'extents':[8], 'bit_strides':[64],
                'dtype_id':DTYPES['u64']['id'], 'alignment':16, 'origin_byte_offset':0, 'origin_bit_offset':0}
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps({'scalars':{'M':96}, 'resources':{
        'input':resource, 'output':{**resource,'data':None,'output':'output.bin'}}}))

    class Backend(MemoryBackend):
        def __init__(self,*args,**kwargs):
            super().__init__([],result)

        def __enter__(self):
            return self

        def __exit__(self,*args):
            assert not self.memory

        def launch(self,params,geometry,stream):
            super().launch(params,geometry,stream)
            # The host-less instruction is consumed by the emulated kernel, not merely copied.
            factor = C.c_uint64.from_address(params.address+blob_offset).value
            residue = StubResidue.from_address(params.address+host['groups'][-1]['residue']['offset'])
            C.c_uint64.from_address(residue.resource).value *= factor

    monkeypatch.setattr(cuda,'CudaBackend',Backend)
    options = {'provider_paths':providers, 'provider_arg_paths':arg_paths, 'host_inputs_path':inputs}
    return path, {'stub.instruction':instruction}, options


def test_launch_mixed_blobs_and_group_providers_return_readback(launch_case):
    path,blobs,options = launch_case
    completed = launcher.launch(path,blobs,**options)
    report = json.loads(completed.stdout)
    actual = bytes.fromhex(report['readback_bytes']['output'])
    assert struct.unpack('<8Q',actual) == (15,1,0,0,0,0,0,0)
    assert path.with_name('output.bin').read_bytes() == actual
    assert report['geometry']['grid'] == [3,1,1]
    assert report['transfer_reports'][0]['status'] == 'PASS'
    assert report['toolchain_provenance']['observations']['driver_identity'] == 'cpu-fixture'


def test_launch_provider_maps_require_params_group_not_stage_alias(launch_case):
    path,blobs,options = launch_case
    options['provider_paths'] = {name.removesuffix('.params'):path for name,path in options['provider_paths'].items()}
    with pytest.raises(LinkError) as caught:
        launcher.launch(path,blobs,**options)
    failure = caught.value.failures[0]
    assert (failure.clause_id,failure.path) == ('L-HOSTABI','providers')
    assert failure.detail['expected'] == ['stub.alpha.params','stub.integration.params','stub.user.params']


@pytest.mark.parametrize('selection',['missing','host-stage','zero-stage'])
def test_launch_blobs_exactly_cover_hostless_nonzero_stages(launch_case,selection):
    path,blobs,options = launch_case
    if selection == 'missing':
        blobs = {}
    else:
        blobs['stub.alpha' if selection=='host-stage' else 'stub.zero'] = next(iter(blobs.values()))
    with pytest.raises(LinkError) as caught:
        launcher.launch(path,blobs,**options)
    failure = caught.value.failures[0]
    assert failure.path == 'blobs'
    assert failure.detail['expected'] == ['stub.instruction']


def test_launch_blob_extent_is_published_size(launch_case):
    path,blobs,options = launch_case
    blobs['stub.instruction'].write_bytes(bytes(7))
    with pytest.raises(LinkError) as caught:
        launcher.launch(path,blobs,**options)
    failure = caught.value.failures[0]
    assert failure.path == 'stub.instruction'
    assert failure.detail['expected'] == 8
    assert failure.detail['actual'] == 7


@pytest.mark.parametrize(('layout','blobs'),[
    (ParamsLayout(16,8,[('stage',0,8,8)]), {'stage':bytes(7)}),
    (ParamsLayout(16,8,[('stage',3,8,8)]), {'stage':bytes(8)}),
    (ParamsLayout(16,8,[('stage',0,8,16)]), {'stage':bytes(8)}),
])
def test_hostless_blob_size_and_alignment_must_match_link(layout,blobs):
    with pytest.raises(LinkError) as caught:
        launcher._assemble_hostless_params(layout,blobs)
    failure = caught.value.failures[0]
    assert (failure.clause_id,failure.path) == ('L-PARAMS','stage')


@pytest.mark.parametrize(('blobs','missing','unknown'),[
    ({}, ['stage'], []),
    ({'stage':bytes(8), 'other':bytes(8)}, [], ['other']),
])
def test_hostless_blob_inventory(blobs,missing,unknown):
    with pytest.raises(LinkError) as caught:
        launcher._assemble_hostless_params(ParamsLayout(8,8,[('stage',0,8,8)]),blobs)
    failure = caught.value.failures[0]
    assert failure.clause_id == 'L-PARAMS'
    assert failure.detail == {'missing':missing, 'unknown':unknown}


def test_hostless_session_binds_allocated_buffers_and_retains_output(tmp_path,monkeypatch):
    result = _result(None,ParamsLayout(40,8,[('zero',0,0,8),('scale',16,24,8)]))
    path = tmp_path / 'link.json'
    path.write_text(json.dumps(result.to_json_obj()))
    control = tmp_path / 'control.cubin'
    control.write_bytes(b'CPU scale kernel')
    alternate = tmp_path / 'alternate.cubin'
    alternate.write_bytes(b'CPU alternative scale image')
    backends = []

    class Backend(MemoryBackend):
        def __init__(self,cubin,symbol,**kwargs):
            super().__init__([],result)
            backends.append(self)
            self.stream_token = len(backends)
            self.launch_streams = []
            self.current_depth = 0

        @contextmanager
        def current(self):
            self.current_depth += 1
            try:
                yield
            finally:
                self.current_depth -= 1

        def __enter__(self):
            return self

        def __exit__(self,*args):
            self.memory.clear()

        def launch(self,params,geometry,stream):
            self.launch_streams.append(stream)
            source,destination,count,factor = struct.unpack_from('<QQIf',params.bytes,16)
            src = (C.c_float*count).from_address(source)
            dst = (C.c_float*count).from_address(destination)
            for index in range(count):
                dst[index] += src[index]*factor

    monkeypatch.setattr(cuda,'CudaBackend',Backend)
    source = np.arange(12,dtype=np.float32).reshape(3,4)[:,::2]
    builds = []

    def params(addresses):
        # Builders encode driver objects (tensor maps), so the session's context must be current.
        assert backends[0].current_depth == 1
        builds.append(dict(addresses))
        return {'zero':b'', 'scale':struct.pack('<QQIf',addresses['input'],addresses['output'],6,2)}

    with launcher.open_hostless(path,buffers={'input':source,'output':((3,2),np.float32)},
                                params=params,grid=(1,1,1),cubin=control) as session:
        np.testing.assert_array_equal(session.read('output'),np.zeros((3,2),dtype=np.float32))
        session.launch()
        first = session.read('output')
        np.testing.assert_array_equal(first,source*2)
        session.launch()
        np.testing.assert_array_equal(session.read('output'),source*4)
        np.testing.assert_array_equal(first,source*2)
        with session.with_image(alternate) as paired:
            assert paired.addresses == session.addresses
            paired.launch()
            np.testing.assert_array_equal(session.read('output'),source*6)
            session.launch()
            np.testing.assert_array_equal(paired.read('output'),source*8)
            assert backends[1].memory == {}
        session.launch()
        np.testing.assert_array_equal(session.read('output'),source*10)
        assert len(builds) == 1
        assert backends[0].launch_streams == [1,1,1,1]
        assert backends[1].launch_streams == [1]
    assert backends[0].memory == {}


def test_cli_buffer_specs_load_arrays_and_preserve_shape_dtype(tmp_path):
    path = tmp_path / 'input.npy'
    array = np.arange(6,dtype=np.int32).reshape(2,3)
    np.save(path,array)
    buffers = cli._parse_buffer_args([f'input={path}','output=64x32:float32','bf16=1024:uint16'])
    np.testing.assert_array_equal(buffers['input'],array)
    assert buffers['output'] == ((64,32),np.dtype('float32'))
    assert buffers['bf16'] == ((1024,),np.dtype('uint16'))


@pytest.mark.parametrize('specs',[
    ['out'], ['=8:float32'], ['out='], ['out=-1:float32'], ['out=2xx3:float32'],
    ['out=2:missing_dtype'], ['out=2:object'], ['out=2:float32','out=4:float32'],
])
def test_cli_buffer_specs_reject_invalid_allocations(specs):
    with pytest.raises(ValueError):
        cli._parse_buffer_args(specs)


def test_cli_cluster_is_one_comma_token_and_does_not_swallow_the_link():
    args = cli._parser().parse_args(['launch', '--cluster', '2,1,1', 'link.json'])
    assert args.link_json == 'link.json'
    assert cli._parse_dimensions(args.cluster, '--cluster') == (2, 1, 1)
    with pytest.raises(ValueError):
        cli._parse_dimensions('2,1', '--cluster')


class ClockBackend:
    stream_token = 7

    def __init__(self):
        self.clock = 0.0
        self.events = {}

    def launch(self,params,geometry,stream):
        self.clock += 0.25

    def event_create(self):
        event = len(self.events)+1
        self.events[event] = None
        return event

    def event_record(self,event,stream):
        self.events[event] = self.clock

    def event_synchronize(self,event):
        pass

    def event_elapsed_time(self,start,end):
        return self.events[end]-self.events[start]

    def event_destroy(self,event):
        del self.events[event]


def test_hostless_timing_measures_whole_batch():
    backend = ClockBackend()
    with launcher.HostlessSession(backend,ExitStack(),{},{},None,None,
                                  kernel_symbol='test',device=0,stream=7) as session:
        assert session.time(12) == 3.0
        assert session.time(4) == 1.0
    assert backend.clock == 4.0
    assert backend.events == {}


@pytest.mark.parametrize('duration',[0.0,-1.0,float('nan'),float('inf')])
def test_hostless_timing_rejects_invalid_event_duration(duration,monkeypatch):
    backend = ClockBackend()
    monkeypatch.setattr(backend,'event_elapsed_time',lambda start,end: duration)
    with launcher.HostlessSession(backend,ExitStack(),{},{},None,None,
                                  kernel_symbol='test',device=0,stream=7) as session:
        with pytest.raises(LinkError) as caught:
            session.time(1)
    assert caught.value.failures[0].clause_id == 'L-LAUNCH'
    assert backend.events == {}


@pytest.mark.parametrize('repetitions',[0,-1,1.5,True])
def test_hostless_timing_requires_positive_integer_batch(repetitions):
    backend = ClockBackend()
    with launcher.HostlessSession(backend,ExitStack(),{},{},None,None,
                                  kernel_symbol='test',device=0,stream=7) as session:
        with pytest.raises(ValueError):
            session.time(repetitions)
    assert backend.clock == 0.0

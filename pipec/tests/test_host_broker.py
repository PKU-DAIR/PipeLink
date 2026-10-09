from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import ctypes as C
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import pytest

from pipec2.canonjson import canonical_digest
from pipec2.host import abi
from pipec2.host import broker
from pipec2.host.cuda import CudaBackend
from pipec2.host.fixture import Domain, Fixture, Resource, symmetric
from pipec2.host.profile import DTYPES, HOST_ABI, dtypes_digest
from pipec2.host.provider import build_provider
from pipec2.model import LinkError, ParamsLayout
from pipec2.symbols import cuda_home

OWNERS = ('alpha', 'integration', 'user')
PROFILE = 'sm_103a'  # the native trio fixture's own target; linked_trio re-stamps per profile
# The user kernel's output after any single launch of a session: gridDim.x for M=96, the
# user workspace counter after this launch's increment (1 only if the workspace was reset
# before the launch; 3 after three launches without reset), six zero lanes.
OUTPUT = (3, 1, 0, 0, 0, 0, 0, 0)

HEADER = '''#pragma once
#include <stdint.h>
typedef struct StubArgs { uint32_t mode; uint32_t workspace_bytes; uint64_t tag; } StubArgs;
typedef struct StubResidue { uint64_t tag; uint64_t resource; uint64_t workspace; uint64_t next_peer; } StubResidue;
'''


class StubResidue(C.Structure):
    _fields_ = [('tag', C.c_uint64), ('resource', C.c_uint64), ('workspace', C.c_uint64), ('next_peer', C.c_uint64)]


# These are native providers, with one private session per attempted build and
# no global provider state or provider-to-provider dependency. The Python caller
# instruments calls only to observe the protocol, not to implement it.
SOURCE = r'''
#include "pipelink_host_abi_v2.h"
#include "stub.h"
#include <new>
#include <cstring>
#define CAT_(a,b) a##b
#define CAT(a,b) CAT_(a,b)
struct plk_session_v2 {
  const plk_common_v2* common;
  const plk_resources_v2* resources;
  const StubArgs* args;
  const StubResidue* residue;
};
static uint64_t scalar(const plk_common_v2* c) {
  const auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(c->shared) + c->shared->record_offset);
  return records[0].value_bits;
}
extern "C" int CAT(pipelink_params_query_,OWNER)(const plk_common_v2* c,const void* raw,plk_demands_v2* d) {
  const auto& a=*static_cast<const StubArgs*>(raw);
  if (a.mode & 2) return PLK_E_ARGUMENT;
  if (c->header.abi_version != 2 || c->header.struct_bytes != sizeof(*c) ||
      d->header.struct_bytes != sizeof(*d) || c->shared->record_count != 1) return PLK_E_CONTRACT;
  const plk_digest_v2 absent = {};
  if (!std::memcmp(&c->profile.dtypes,&absent,sizeof(absent))) return PLK_E_CONTRACT;
  d->channel_count=d->channel_capacity;
  for(unsigned i=0;i<d->channel_count;++i) {
    const auto& m=c->channels[i]; auto& out=d->channels[i];
    if(m.channel_id!=out.channel_id || m.rank!=out.rank) return PLK_E_CONTRACT;
    out.dtype_id=m.dtype_id; out.access=m.access;
    // A world-sized demand (one 64-byte shard per rank), sized from the owner's first declared domain.
    out.minimum_allocation_bytes=64ull*(c->domain_count ? c->domains[0].world : 1); out.alignment_bytes=16;
    for(unsigned j=0;j<m.rank;++j) { out.extents[j]=m.extents[j]; out.bit_strides[j]=m.bit_strides[j]; }
  }
  d->workspace_count=d->workspace_capacity;
  d->workspaces[0].bytes=a.workspace_bytes;
  d->workspaces[0].alignment_bytes=WORKSPACE_ALIGN;
  if (AUTHORITY || (a.mode & 16)) {
    d->geometry.geometry_present=1; d->geometry.launch_required=1;
    d->geometry.block=c->block; d->geometry.cluster=c->cluster;
    d->geometry.dynamic_smem_bytes=c->dynamic_smem_bytes;
    d->geometry.grid={static_cast<uint32_t>(((scalar(c)+31)/32)*c->cluster.x),c->cluster.y,c->cluster.z};
    if(a.mode&16) d->geometry.grid.x+=c->cluster.x;
  }
  if(a.mode&32) ++d->channel_count;
  if(a.mode&64) d->channels[0].extents=nullptr;
  if(a.mode&128) ++d->workspaces[0].workspace_id;
  if(a.mode&8192) d->channels[0].minimum_allocation_bytes=128;
  return PLK_OK;
}
extern "C" int CAT(pipelink_params_build_,OWNER)(const plk_common_v2* c,const void* raw,
    const plk_resources_v2* r,void* out,size_t cap,plk_transfer_report_v2* transfers,
    uint32_t transfer_capacity,plk_session** session) {
  const auto& a=*static_cast<const StubArgs*>(raw);
  if(cap!=sizeof(StubResidue) || !out || *session || r->shared!=c->shared ||
     r->workspace_count!=1 || r->channel_count!=1 || bool(transfer_capacity)!=bool(AUTHORITY) ||
     r->geometry.geometry_present!=1 || !r->cuda_stream_token) return PLK_E_CONTRACT;
  if(r->workspaces[0].bytes!=a.workspace_bytes ||
     (a.workspace_bytes==0 && r->workspaces[0].device_address!=0) ||
     r->workspaces[0].device_address%WORKSPACE_ALIGN) return PLK_E_CONTRACT;
  auto* residue=static_cast<StubResidue*>(out);
  // A peer reader encodes the next domain rank's base (a nonlocal table entry); the domain's
  // multicast base when it has no table; local channels have neither.
  const auto& ch=r->channels[0];
  const plk_domain_v2* dom=plk_find_domain(c,ch.domain_id);
  if(ch.domain_id && (!dom || dom!=&c->domains[0])) return PLK_E_CONTRACT;
  const uint64_t* peers=ch.peer_addresses;
  *residue={a.tag,ch.allocation_address,r->workspaces[0].device_address,
            peers ? peers[(dom->rank+1)%dom->world] : ch.multicast_address};
  if(!(a.mode&1)) {
    *session=new(std::nothrow) plk_session_v2{c,r,&a,residue};
    if(!*session) return PLK_E_RESOURCE;
  }
  for(uint32_t i=0;i<transfer_capacity;++i) {
    static const plk_tma_tiled_detail_v2 correct = {3,1,{8},{1},0};
    static const plk_tma_tiled_detail_v2 wrong = {2,1,{8},{1},0};
    auto& t=transfers[i];
    if(t.transfer_id!=9+i) return PLK_E_CONTRACT;
    t.access=PLK_ACCESS_READ;
    t.residue_offset=8; t.representation_bytes=8; t.representation_align=8;
    t.storage_dtype_id=c->channels[0].dtype_id;
    t.ring_depth=2; t.ring_stage_stride_bytes=64;
    t.footprint_bytes_per_stage=64/transfer_capacity;
    t.credit_bytes_per_stage=((a.mode&1024)?64:128)/transfer_capacity;
    t.source_origin_bits=0; t.source_span_bits=(a.mode&4096)?520:512;
    t.vendor_kind=PLK_TRANSFER_TMA_TILED;
    t.vendor_detail=(a.mode&2048)?&wrong:&correct;
  }
  if(a.mode&4) return PLK_E_BUILD;
  if(a.mode&512) const_cast<plk_resources_v2*>(r)->geometry.grid.x++;
  return PLK_OK;
}
extern "C" int CAT(pipelink_params_release_,OWNER)(plk_session* session) {
  if(!session) return PLK_OK;
  // The common image, resource records, residue and owner args outlive release.
  const auto& s=*session;
  bool valid=s.common->header.abi_version==2 && scalar(s.common)>0 &&
      s.resources->shared==s.common->shared && s.residue->tag==s.args->tag;
  bool error=s.args->mode&8;
  delete session;
  return valid && !error ? PLK_OK : PLK_E_RELEASE;
}
'''


def _make_trio(root, profile=PROFILE):
    """Native C++ owners and the exact resolved host publication consumed by the broker."""
    header = root / 'stub.h'
    header.write_text(HEADER)
    source = root / 'provider.cpp'
    source.write_text(SOURCE)
    extra = {'StubArgs': ['mode', 'workspace_bytes', 'tag'], 'StubResidue': ['tag', 'resource', 'workspace', 'next_peer']}
    types = {'StubArgs': 1, 'StubResidue': 2}
    groups, paths, cursor = [], {}, 40
    field = {'id': 7, 'name': 'stub.M', 'type': 'u64', 'source': {'kind': 'caller', 'name': 'M'}}
    schema = {'required': {'M': {'kind': 'runtime', 'type': 'i64'}}, 'optional': {}, 'pinned': {},
              'runtime_i32': [], 'hardware_rules': {}, 'allow_no_launch': False}
    scheduler = {'owner': 'stub.integration', 'strategy': 'stub.grid', 'schema': schema,
                 'input_schema_digest': canonical_digest(schema),
                 'inputs': [{'name': 'M', 'source': {'kind': 'caller', 'name': 'M'}}],
                 'runtime_inputs': ['M'], 'workspace_ids': []}
    target = {'dtypes': dtypes_digest(profile)}
    for i, owner in enumerate(OWNERS):
        name, group = f'stub.{owner}', f'stub.{owner}.params'
        contract = canonical_digest({'owner': name, 'fixture': 'three-independent-native-providers'})
        output = root / f'{owner}.so'
        align = (64, 256, 32)[i]
        channel_name = 'stub.output' if owner == 'user' else 'stub.input'
        inventory = build_provider([source], output, owner=owner, owner_contract_digest=contract,
            profile=profile, dtypes_digest=target['dtypes'], type_ids=types, extra_headers=[header], extra_types=extra,
            compile_flags=[f'-I{root}', f'-I{abi.header_path().parent}', f'-DOWNER={owner}',
                           f'-DAUTHORITY={int(owner=="integration")}', f'-DWORKSPACE_ALIGN={align}'])
        paths[group] = output
        shared = {'id': 1, 'fields': [field], 'read_roles': [f'{name}:compute'], 'offset': 0,
                  'bytes': 40, 'align': 8}
        residue_type = next(row for row in inventory if row['name']=='StubResidue')
        residue = {'id': 2, 'name': name+'.residue', 'bytes': 32, 'align': 8, 'type_id': types['StubResidue'],
                   'layout_digest': residue_type['layout_digest'], 'read_roles': [f'{name}:compute'], 'offset': cursor}
        cursor += 32
        channel = {'id': 20+i, 'name': channel_name, 'logical_dtype': 'u64', 'storage_dtype': 'u64',
            'dtype_id': DTYPES['u64']['id'], 'rank': 1,
            'access': 'write' if owner=='user' else 'read', 'domain': None, 'views': [],
            'source': {'kind': 'caller', 'name': 'output' if owner=='user' else 'input'},
            'initializer': {'kind': 'caller'}, 'alias': None, 'lifetime': 'release'}
        workspace = {'id': 40+i, 'name': name+'.workspace', 'params_group': group, 'align': align, 'space': 'gmem',
                     'channel': None, 'bytes': 'query',
                     'initializer': {'kind': 'zero' if owner == 'user' else 'none'},
                     'reset': 'launch' if owner == 'user' else 'open'}
        transfer = {'id': 9, 'name': 'stub.integration.load', 'channel': channel_name,
                    'device_endpoint': {'stage': name, 'endpoint': 'tile'}, 'access': 'read',
                    'endpoint': {'space': 'smem', 'dtype': 'u64', 'elem_bits': 64,
                                 'tile': {'dtype': 'u64', 'shape': [8]},
                                 'layout': {'encoding': 'pl-layout-v2', 'elem_bits': 64, 'shape': [8], 'stride': [1]},
                                 'swizzle': {'B': 3, 'M': 1, 'S': 3, 'unit': 'element'},
                                 'ring': {'depth': 2, 'stage_stride_bytes': 64},
                                 'align': 16, 'tile_bytes': 64, 'access': 'produce'},
                    'expect_tx_bytes': 128}
        groups.append({'owner': name, 'target': target, 'owner_id': i+1, 'params_group': group, 'stage_ids': [name],
            'provider': {'protocol': 2, **{phase: f'pipelink_params_{phase}_{owner}' for phase in ('abi','query','build','release')},
                         'owner_contract_digest': contract, 'abi_types': inventory,
                         'args': {'bytes':16, 'align':8, 'type_id':types['StubArgs']}},
            'scalars': shared, 'residue': residue, 'domains': [], 'channels': [channel], 'workspaces': [workspace],
            'transfers': [transfer] if owner=='integration' else [], 'published_facts': [],
            'scheduler_contract': scheduler if owner=='integration' else {'owner': 'stub.integration'}})
    host = {'protocol':2, 'host_abi':HOST_ABI, 'profile':profile, 'target':target,
            'groups':groups, 'params': {'bytes':cursor,'align':8,'blobs':[]}, 'facts':{}, 'bindings':{},
            'scheduler_contract': scheduler, 'role_views': {}, 'role_view_header': '',
            'fact_inventory': {}, 'declarations': [], 'variants': {}, 'static_launch': {},
            'selection_digest': canonical_digest(groups), 'toolchain_provenance': {}}
    result = SimpleNamespace(host=host, params=ParamsLayout(cursor,8,[]), threads=32,
                             dyn_smem_bytes=0, cluster=(1,1,1))
    return result, paths, root


@pytest.fixture(scope='module')
def trio(tmp_path_factory):
    return _make_trio(tmp_path_factory.mktemp('host-native-trio'))


def linked_trio(trio, root, profile=PROFILE):
    """Compile a kernel consuming the same published params offsets as the CPU backend."""
    from pipec2.model import LinkResult
    baseline, paths, _ = _make_trio(root, profile)
    host = baseline.host
    size = baseline.params.total_bytes
    source = f'struct Params {{ unsigned long long words[{size // 8}]; }};\n'
    source += 'extern "C" __global__ void pipelink_kernel(Params p) {\n'
    for i, group in enumerate(host['groups']):
        word = group['residue']['offset'] // 8
        source += f'  if(p.words[{word}] != {100+i}) asm volatile("trap;");\n'
    word = host['groups'][-1]['residue']['offset'] // 8
    source += f'''
  if(blockIdx.x==0 && threadIdx.x<8) {{
    auto* output=reinterpret_cast<unsigned long long*>(p.words[{word+1}]);
    auto* counter=reinterpret_cast<unsigned long long*>(p.words[{word+2}]);
    output[threadIdx.x]=threadIdx.x==0 ? gridDim.x : threadIdx.x==1 ? atomicAdd(counter,1ull)+1 : 0;
  }}
}}
'''
    result=LinkResult('pipelink_kernel',32,32,0,(1,1,1),baseline.params,{},{},{},[],
                      (0,0,0),{},{},None,host)
    return result,paths,source


class MemoryBackend:
    """CPU memory execution of the broker's real native provider protocol.

    ``launch`` emulates the linked trio's user kernel on host memory: the
    workspace counter is found through the user residue in the params image.
    """
    stream_token = 7
    observations = {'observed_sm_count': 148, 'max_grid':(2147483647,65535,65535),
                    'max_block':(1024,1024,64), 'max_threads':1024,
                    'arch': 'sm103', 'toolkit': 'cuda-13.3.0', 'driver_identity': 'cpu-fixture'}
    lost = False

    def __init__(self, events, result):
        self.events, self.memory, self.result = events, {}, result
        self.fail_launch = self.fail_sync = False

    @contextmanager
    def current(self):
        yield

    def allocate(self, size, align):
        self.events.append(('allocate',size,align))
        if not size:
            return 0
        buffer = abi.aligned_buffer(size,align)
        self.memory[buffer.address] = buffer
        return buffer.address

    def free(self, address):
        self.events.append(('free',address))
        if address:
            del self.memory[address]

    def upload(self, address, data):
        C.memmove(address,data,len(data))

    def zero(self, address, size, stream):
        self.events.append(('zero',stream))
        if size:
            C.memset(address,0,size)

    def launch(self, params, geometry, stream):
        self.events.append(('launch',broker._dims(geometry.grid),stream))
        if self.fail_launch:
            raise RuntimeError('synthetic launch error')
        group = self.result.host['groups'][-1]
        residue = StubResidue.from_address(params.address+group['residue']['offset'])
        counter = C.c_uint64.from_address(residue.workspace)
        counter.value += 1
        (C.c_uint64*8).from_address(residue.resource)[:] = [geometry.grid.x,counter.value,0,0,0,0,0,0]

    def synchronize(self):
        self.events.append(('synchronize',))
        if self.fail_sync:
            raise RuntimeError('synthetic context loss')

    def abandon(self):
        self.events.append(('abandon',))
        self.lost = True

    def readback(self, address, size):
        return C.string_at(address,size)


def instrument(monkeypatch, events):
    original = broker._load_owner
    def load(*args, **kwargs):
        owner = original(*args, **kwargs)
        for phase, function in list(owner.functions.items()):
            def call(*values, phase=phase, function=function, name=owner.name):
                events.append((phase,name))
                return function(*values)
            owner.functions[phase] = call
        return owner
    monkeypatch.setattr(broker,'_load_owner',load)
    close = broker.dlclose
    monkeypatch.setattr(broker,'dlclose',lambda handle: (events.append(('dlclose',)),close(handle))[1])


def owner_args(modes=None):
    modes = modes or {}
    return {f'stub.{owner}.params':struct.pack('<IIQ',modes.get(owner,0),[13,0,65][i],100+i)
            for i,owner in enumerate(OWNERS)}


def invocation(trio, monkeypatch, *, modes=None, shape=96, result=None):
    baseline, paths, root = trio
    result = deepcopy(baseline) if result is None else result
    events = []
    instrument(monkeypatch,events)
    backend = MemoryBackend(events,result)
    input_buffer, output_buffer = abi.aligned_buffer(64,16), abi.aligned_buffer(64,16)
    resources = {
        'input': Resource(input_buffer.address,64,(8,),(64,),DTYPES['u64']['id'],16,lender=input_buffer),
        'output':Resource(output_buffer.address,64,(8,),(64,),DTYPES['u64']['id'],16,lender=output_buffer),
    }
    return result, paths, owner_args(modes), resources, {'M':shape}, backend, events


def launch(session, backend, resources, events, stream):
    """One session launch: exactly its reset and launch on ``stream``; the caller drains and reads back."""
    start = len(events)
    session.launch(stream)
    assert events[start:] == [('zero',stream),('launch',session.geometry['grid'],stream)]
    backend.synchronize()
    return struct.unpack('<8Q',backend.readback(resources['output'].address,64))


def releases(events):
    return [event[1] for event in events if event[0]=='release']


def test_native_trio_session_opens_once_resets_per_launch_and_releases_in_reverse(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'user':1})
    with broker.open_host(result,paths,args,resources,scalars,backend) as session:
        assert [event[0] for event in events] == ['abi']*3+['query']*3+['allocate','zero','build','build','build','synchronize']
        assert events[7] == ('zero',backend.stream_token)
        assert [event[1] for event in events if event[0]=='build'] == ['stub.alpha','stub.integration','stub.user']
        assert session.geometry['grid'] == (3,1,1) and session.geometry['launch_required']
        [base] = backend.memory
        for _ in range(3):
            assert launch(session,backend,resources,events,11) == OUTPUT
        closing = len(events)
    assert events[closing:] == [('synchronize',),('release','stub.user'),('release','stub.integration'),
                                ('release','stub.alpha'),('free',base)]+[('dlclose',)]*3
    assert not backend.memory
    session.close()
    assert len(events) == closing+8




@pytest.mark.parametrize('shape,cluster,grid',[(17,(1,1,1),(1,1,1)),(96,(2,1,1),(6,1,1))])
def test_query_geometry_varies_with_original_shape_and_cluster(trio,monkeypatch,shape,cluster,grid):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,shape=shape)
    result.cluster=cluster
    with broker.open_host(result,paths,args,resources,scalars,backend) as session:
        assert session.geometry['grid']==grid
        assert launch(session,backend,resources,events,11)==(grid[0],1)+(0,)*6


def test_abi_mismatch_rejects_all_queries(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    result.host['groups'][1]['provider']['owner_contract_digest']='f'*64
    with pytest.raises(LinkError,match='L-HOSTABI'):
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert events==[('abi','stub.alpha'),('abi','stub.integration')]+[('dlclose',)]*3

@pytest.mark.parametrize('type_name', ['plk_header_v2', 'StubResidue'])
def test_native_report_rejects_compiled_layout_mismatch(trio, type_name):
    result, paths, _ = trio
    group = deepcopy(result.host['groups'][0])
    owner = broker._load_owner(group, paths[group['params_group']], owner_args()[group['params_group']])
    try:
        broker._report(owner, group['target'])
        row = next(row for row in group['provider']['abi_types'] if row['name'] == type_name)
        native_digest = row['layout_digest']
        row['layout_digest'] = ('0' if native_digest[0] != '0' else '1') + native_digest[1:]
        with pytest.raises(LinkError) as caught:
            broker._report(owner, group['target'])
        failure = caught.value.failures[0]
        assert failure.clause_id == 'L-HOSTABI'
        assert failure.path == f"{group['owner']}:{type_name}"
        assert failure.detail['expected'][3] == row['layout_digest']
        assert failure.detail['actual'][3] == native_digest
    finally:
        broker.dlclose(owner.library._handle)


@pytest.mark.parametrize('change', ['missing', 'renumbered'])
def test_publication_requires_complete_pipec_neutral_ids(trio, monkeypatch, change):
    result, paths, args, resources, scalars, backend, _ = invocation(trio, monkeypatch)
    group = result.host['groups'][0]
    inventory = group['provider']['abi_types']
    row = next(row for row in inventory if row['name'] == 'plk_header_v2')
    if change == 'missing':
        inventory.remove(row)
    else:
        row['id'] = 99
    with pytest.raises(LinkError) as caught:
        broker.open_host(result, paths, args, resources, scalars, backend)
    failure = caught.value.failures[0]
    assert failure.clause_id == 'L-HOSTABI'
    assert failure.path == f"{group['owner']}:plk_header_v2"


@pytest.mark.parametrize('type_ids', [
    {'StubArgs': 1},
    {'StubArgs': 1, 'StubResidue': 1},
    {'StubArgs': 1, 'StubResidue': 2, 'plk_header_v2': 99},
    {'StubArgs': 0x70000000, 'StubResidue': 2},
    {'StubArgs': 0x7000000A, 'StubResidue': 2},
    {'StubArgs': 0x700000FF, 'StubResidue': 2},
], ids=['missing-local', 'duplicate-local-id', 'explicit-neutral', 'reserved-start',
        'neutral-id-collision', 'reserved-end'])
def test_provider_rejects_invalid_owner_type_inventory(tmp_path, type_ids):
    header = tmp_path / 'stub.h'
    header.write_text(HEADER)
    with pytest.raises(ValueError):
        build_provider([], tmp_path / 'invalid.so', owner='invalid',
            owner_contract_digest=canonical_digest({'owner': 'invalid'}),
            profile=PROFILE, dtypes_digest=dtypes_digest(PROFILE), type_ids=type_ids,
            extra_headers=[header],
            extra_types={'StubArgs': ['mode', 'workspace_bytes', 'tag'],
                         'StubResidue': ['tag', 'resource', 'workspace', 'next_peer']})



def test_native_report_rejects_wrong_dtype_contract(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    original = broker._load_owner
    def load(*args, **kwargs):
        owner = original(*args, **kwargs)
        report = owner.functions['abi']
        def changed(pointer):
            status = report(pointer)
            value = C.cast(pointer,C.POINTER(abi.plk_abi_report_v2)).contents
            value.profile.dtypes.sha256[:] = bytes(32)
            return status
        owner.functions['abi'] = changed
        return owner
    monkeypatch.setattr(broker,'_load_owner',load)
    with pytest.raises(LinkError) as caught:
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert caught.value.failures[0].clause_id == 'L-HOSTABI'
    assert caught.value.failures[0].detail['contract'] == 'dtypes'
    assert events == [('abi','stub.alpha')]+[('dlclose',)]*3


@pytest.mark.parametrize('mode,reason',[(16,'geometry'),(32,'inventory'),(64,'inventory'),(128,'inventory'),(8192,'resource_range')])
def test_query_contract_rejections_precede_allocation_and_build(trio,monkeypatch,mode,reason):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'alpha':mode})
    with pytest.raises(LinkError) as failure:
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert failure.value.failures[0].detail['subreason']==reason
    assert not any(event[0] in ('allocate','zero','build','release','launch') for event in events)


@pytest.mark.parametrize('mode',[4,5])
def test_failed_build_releases_partial_or_null_session_after_drain(trio,monkeypatch,mode):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'integration':mode})
    with pytest.raises(RuntimeError,match='integration build failed'):
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert [event[0] for event in events] == ['abi']*3+['query']*3+['allocate','zero','build','build','synchronize',
                                                                     'release','release','free']+['dlclose']*3
    assert [event[1] for event in events if event[0]=='build']==['stub.alpha','stub.integration']
    assert releases(events)==['stub.integration','stub.alpha']
    assert not backend.memory


def test_build_cannot_override_authoritative_query_geometry(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'integration':512})
    with pytest.raises(LinkError,match='override query geometry'):
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert not any(event[0]=='launch' for event in events)
    assert releases(events)==['stub.integration','stub.alpha']
    assert not backend.memory


def test_native_query_failure_remains_explicit_and_resource_free(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'integration':2})
    with pytest.raises(RuntimeError,match='integration query failed with status 3'):
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert not any(event[0] in ('allocate','zero','build','release','launch') for event in events)


def test_release_errors_do_not_skip_remaining_owners(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'user':8,'integration':8})
    with pytest.raises(ExceptionGroup,match='host release/cleanup failed') as failure:
        with broker.open_host(result,paths,args,resources,scalars,backend) as session:
            assert launch(session,backend,resources,events,11) == OUTPUT
    assert len(failure.value.exceptions)==2
    assert releases(events)==['stub.user','stub.integration','stub.alpha']
    assert not backend.memory


def test_failed_launch_leaves_the_session_open_until_close_releases(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    session = broker.open_host(result,paths,args,resources,scalars,backend)
    backend.fail_launch = True
    opened = len(events)
    with pytest.raises(RuntimeError,match='synthetic launch error'):
        session.launch(11)
    assert events[opened:] == [('zero',11),('launch',(3,1,1),11)]
    [base] = backend.memory
    session.close()
    assert events[opened+2:] == [('synchronize',),('release','stub.user'),('release','stub.integration'),
                                 ('release','stub.alpha'),('free',base)]+[('dlclose',)]*3
    assert not backend.memory


def test_undrainable_context_never_frees_inflight_storage(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    session = broker.open_host(result,paths,args,resources,scalars,backend)
    backend.fail_launch = True
    with pytest.raises(RuntimeError,match='synthetic launch error'):
        session.launch(11)
    backend.fail_sync = True
    closing = len(events)
    with pytest.raises(ExceptionGroup,match='host release/cleanup failed') as failure:
        session.close()
    assert [str(exc) for exc in failure.value.exceptions] == ['synthetic context loss']
    assert events[closing:] == [('synchronize',),('abandon',),('release','stub.user'),('release','stub.integration'),
                                ('release','stub.alpha')]+[('dlclose',)]*3
    assert backend.lost and backend.memory


def _domain(result, owner_domains, **channels):
    """Declare owner domains ({owner: [(id, name)]}) and channel (domain, views) by channel name."""
    for group in result.host['groups']:
        group['domains'] = [{'id': i, 'name': n} for i, n in owner_domains.get(group['owner'], [])]
        for channel in group['channels']:
            if channel['name'] in channels:
                channel['domain'], channel['views'] = channels[channel['name']]


def _residues(session, result):
    return [StubResidue.from_address(session.params.address + group['residue']['offset'])
            for group in result.host['groups']]


def test_owners_receive_their_own_domain_tables_and_declared_views(trio, monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    # Readers depend on `tp` through the shared peer channel; the user owner on `dp` only (no channel).
    _domain(result, {'stub.alpha': [(3, 'tp')], 'stub.integration': [(5, 'tp')], 'stub.user': [(1, 'dp')]},
            **{'stub.input': ('tp', ['peer'])})
    tp, dp, u64 = Domain(1, 3), Domain(0, 2), DTYPES['u64']['id']
    tensor = dict(extents=(8,), bit_strides=(64,), dtype_id=u64, alignment=16)
    with Fixture(backend) as rank0, Fixture(backend) as rank1, Fixture(backend) as rank2:
        fixtures = [rank0, rank1, rank2]
        shards = symmetric(fixtures, 'input', domain='tp', data=[bytes(64 * 3)] * 3, allocation_bytes=64 * 3, **tensor)
        assert [(shard.domain, shard.lender) for shard in shards] == [('tp', f) for f in fixtures]
        # The user owner's query sizes its local output by ITS domain (dp world 2), not by tp.
        short = rank1.tensor('short', data=None, allocation_bytes=64, output=True, **tensor)
        output = rank1.tensor('output', data=None, allocation_bytes=128, output=True, **tensor)
        domains = {'tp': tp, 'dp': dp}
        with pytest.raises(LinkError) as caught:
            broker.open_host(result,paths,args,{'input': shards[1], 'output': short},scalars,backend,domains=domains)
        assert (caught.value.failures[0].detail['subreason'], caught.value.failures[0].path) == (
            'resource_range', 'stub.output')
        loans = {'input': shards[1], 'output': output}
        with broker.open_host(result,paths,args,loans,scalars,backend,domains=domains) as session:
            alpha, integration, user = _residues(session, result)
            assert alpha.next_peer == integration.next_peer == shards[2].address
            assert user.next_peer == 0
            assert launch(session,backend,loans,events,11) == OUTPUT
    # A multicast-only channel receives the multicast base and no peer table, even from a
    # resource that also carries one.
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    _domain(result, {'stub.alpha': [(3, 'tp')], 'stub.integration': [(5, 'tp')]},
            **{'stub.input': ('tp', ['multicast'])})
    lender = abi.aligned_buffer(128, 16)  # 64 bytes per rank of a world-2 domain
    resources['input'] = replace(resources['input'], address=lender.address, allocation_bytes=128, domain='tp',
                                 peers=(lender.address, 1 << 20), multicast=1 << 30, lender=lender)
    with broker.open_host(result,paths,args,resources,scalars,backend,domains={'tp': Domain(0, 2)}) as session:
        alpha, integration, _ = _residues(session, result)
        assert alpha.next_peer == integration.next_peer == 1 << 30


@pytest.mark.parametrize('change,reason,path', [
    ('missing-domain', 'inventory', 'domains'),
    ('rank-outside-world', 'domain', 'domains.tp'),
    ('local-resource', 'domain', 'stub.input'),
    ('other-domain', 'domain', 'stub.input'),
    ('table-shorter-than-world', 'domain', 'stub.input'),
    ('table-misses-local-address', 'domain', 'stub.input'),
    ('misaligned-peer', 'resource_range', 'stub.input.peers[1]'),
    ('null-peer', 'resource_range', 'stub.input.peers[1]'),
    ('missing-multicast', 'domain', 'stub.input'),
    ('misaligned-multicast', 'resource_range', 'stub.input.multicast'),
    ('local-channel-symmetric-resource', 'domain', 'stub.output'),
])
def test_domain_resource_rejections_precede_query(trio, monkeypatch, change, reason, path):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    _domain(result, {'stub.alpha': [(3, 'tp')], 'stub.integration': [(5, 'tp')]},
            **{'stub.input': ('tp', ['multicast', 'peer'])})
    address, other = resources['input'].address, 1 << 20
    loan = dict(domain='tp', peers=(address, other), multicast=1 << 30)
    loan.update({'local-resource': dict(domain=None, peers=(), multicast=0), 'other-domain': dict(domain='dp'),
                 'table-shorter-than-world': dict(peers=(address,)), 'table-misses-local-address': dict(peers=(other, address)),
                 'misaligned-peer': dict(peers=(address, other + 8)), 'null-peer': dict(peers=(address, 0)),
                 'missing-multicast': dict(multicast=0),
                 'misaligned-multicast': dict(multicast=(1 << 30) + 8)}.get(change, {}))
    resources['input'] = replace(resources['input'], **loan)
    if change == 'local-channel-symmetric-resource':
        resources['output'] = replace(resources['output'], domain='tp', peers=(resources['output'].address, other))
    domains = {} if change == 'missing-domain' else {'tp': Domain(2 if change == 'rank-outside-world' else 0, 2)}
    with pytest.raises(LinkError) as caught:
        broker.open_host(result,paths,args,resources,scalars,backend,domains=domains)
    failure = caught.value.failures[0]
    assert (failure.clause_id, failure.detail['subreason'], failure.path) == ('L-HOSTCHAN', reason, path)
    assert not any(event[0] in ('query', 'allocate', 'build') for event in events)


def test_caller_alias_and_published_range_reject_before_build(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    resources['output']=replace(resources['output'],address=resources['input'].address,lender=resources['input'].lender)
    with pytest.raises(LinkError,match='overlapping'):
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert not any(event[0] in ('build','release','launch') for event in events)
    for group in result.host['groups']:
        group['scalars']['offset']=result.params.total_bytes
    with pytest.raises(LinkError,match='published region'):
        broker.open_host(result,paths,args,resources,scalars,backend)


@pytest.mark.parametrize('mode,check,expected,actual', [
    (1024, 'credit_bytes_per_stage', 128, 64),
    (2048, 'swizzle', 3, 2),
    (4096, 'source_span', {'maximum_end_bits': 512}, {'origin_bits': 0, 'end_bits': 520}),
])
def test_transfer_report_rejected_and_attempted_owners_released(trio,monkeypatch,mode,check,expected,actual):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch,modes={'integration':mode})
    with pytest.raises(LinkError) as caught:
        broker.open_host(result,paths,args,resources,scalars,backend)
    failure = caught.value.failures[0]
    assert (failure.clause_id, failure.path) == ('L-HOSTCHAN', 'stub.integration.load')
    assert failure.detail == {'subreason': 'transfer', 'check': check, 'expected': expected, 'actual': actual}
    assert releases(events) == ['stub.integration','stub.alpha']
    assert not backend.memory


def test_correct_transfer_report_publishes_pass_and_observed_provenance(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    with broker.open_host(result,paths,args,resources,scalars,backend) as session:
        report = session.transfer_reports[0]
        assert report['status'] == 'PASS'
        assert report['footprint_bytes_per_stage'] == 64
        assert report['credit_bytes_per_stage'] == 128
        assert report['vendor_detail']['swizzle'] == 3
        assert session.toolchain_provenance['observations'] == backend.observations
        assert launch(session,backend,resources,events,11) == OUTPUT


def test_endpoint_accounting_sums_all_declared_transfers(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    transfers = result.host['groups'][1]['transfers']
    transfers.append({**deepcopy(transfers[0]),'id':10,'name':'stub.integration.load.second'})
    with broker.open_host(result,paths,args,resources,scalars,backend) as session:
        assert [row['transfer_id'] for row in session.transfer_reports] == [9,10]
        assert [row['status'] for row in session.transfer_reports] == ['PASS','PASS']
        assert sum(row['footprint_bytes_per_stage'] for row in session.transfer_reports) == 64
        assert sum(row['credit_bytes_per_stage'] for row in session.transfer_reports) == 128
        assert launch(session,backend,resources,events,11) == OUTPUT


def test_workspace_slices_do_not_overlap_and_reset_only_launch_workspace(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    with broker.open_host(result,paths,args,resources,scalars,backend) as session:
        alpha, integration, user = [
            StubResidue.from_address(session.params.address + group['residue']['offset'])
            for group in result.host['groups']]
        assert integration.workspace == 0
        assert alpha.workspace % 64 == user.workspace % 32 == 0
        assert alpha.workspace + 13 <= user.workspace
        C.memset(alpha.workspace,0x5a,13)
        assert launch(session,backend,resources,events,11) == OUTPUT
        assert C.string_at(alpha.workspace,13) == b'Z'*13


def test_declared_alias_uses_runtime_origin_and_reduced_expressions(trio,monkeypatch):
    result,paths,args,resources,scalars,backend,events = invocation(trio,monkeypatch)
    storage = abi.aligned_buffer(128,16)
    resources['input'] = replace(resources['input'],address=storage.address,allocation_bytes=128,lender=storage)
    resources['output'] = replace(resources['input'],origin_byte_offset=16)
    result.host['facts']['offset'] = 16
    result.host['bindings']['zero'] = (0,)
    channel = result.host['groups'][-1]['channels'][0]
    channel['alias'] = {'channel':'stub.input', 'byte_offset': {
        'op':'add','args':[{'op':'mul','args':[{'scalar':'stub.M'},{'binding':'zero'}]},{'fact':'offset'}]},
        'bit_offset':0}
    with broker.open_host(result,paths,args,resources,scalars,backend) as session:
        assert launch(session,backend,resources,events,11) == OUTPUT
    channel['alias']['bit_offset'] = 1
    with pytest.raises(LinkError) as caught:
        broker.open_host(result,paths,args,resources,scalars,backend)
    assert caught.value.failures[0].detail['subreason'] == 'alias'


def gpu_linked_trio(trio, tmp_path, profile, device):
    """Skip unless one physical GPU of ``device`` is selected; then link the trio and build its cubin there."""
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not physical.isdecimal():
        pytest.skip('GPU context proof requires CUDA_VISIBLE_DEVICES=<one physical index>')
    try:
        cuda = cuda_home()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if not (cuda / "bin" / "nvcc").is_file():
        pytest.skip("configured CUDA compiler is unavailable")
    from pipec2.profile import get_profile
    query = subprocess.run(['nvidia-smi','-i',physical,'--query-gpu=index,uuid,compute_cap,utilization.gpu,memory.used',
                            '--format=csv,noheader'],check=True,capture_output=True,text=True)
    observed = 'sm' + query.stdout.split(',')[2].strip().replace('.', '')
    if observed != device:
        pytest.skip(f'GPU {physical} is {observed}; {profile} links run on {device}')
    result,paths,source=linked_trio(trio,tmp_path,profile)
    assert result.host['profile']==profile
    kernel=tmp_path/'kernel.cu'
    kernel.write_text(source)
    subprocess.run([str(cuda/'bin/nvcc'),
                    '-std=c++17',f'-arch={get_profile(profile).arch.sm}','-cubin',f'-I{abi.header_path().parent}',
                    str(kernel),'-o',str(tmp_path/'pipeline.cubin')],
                   check=True,capture_output=True,text=True)
    return result,paths


@pytest.mark.parametrize('profile,device', [('sm_103a', 'sm103'), ('sm_90a', 'sm90'), ('sm_120a', 'sm120')])
def test_production_cli_trio_on_gpu_query_launch_and_readback(trio,tmp_path,profile,device):
    result,paths=gpu_linked_trio(trio,tmp_path,profile,device)
    from pipec2.canonjson import canonical_dumps
    (tmp_path/'link.json').write_text(canonical_dumps(result.to_json_obj()))
    (tmp_path/'input.bin').write_bytes(struct.pack('<8Q',*range(8)))
    resource={'data':'input.bin','allocation_bytes':64,'extents':[8],'bit_strides':[64],
              'dtype_id':DTYPES['u64']['id'],'alignment':16,
              'origin_byte_offset':0,'origin_bit_offset':0}
    inputs={'scalars':{'M':96},'resources':{'input':resource,
            'output':{**resource,'data':None,'output':'output.bin'}}}
    (tmp_path/'inputs.json').write_text(json.dumps(inputs))
    command=['uv','run','--project','pipec','pipec2','launch',str(tmp_path/'link.json'),
             '--host-inputs',str(tmp_path/'inputs.json')]
    for name,packed in owner_args({'user':1}).items():
        args=tmp_path/f'{name}.args'
        args.write_bytes(packed)
        command.extend(['--provider',f'{name}={paths[name]}','--provider-args',f'{name}={args}'])
    completed=subprocess.run(command,capture_output=True,text=True)
    assert completed.returncode==0, completed.stdout+completed.stderr
    assert struct.unpack('<8Q',(tmp_path/'output.bin').read_bytes())==OUTPUT


@pytest.mark.parametrize('profile,device', [('sm_103a', 'sm103'), ('sm_90a', 'sm90'), ('sm_120a', 'sm120')])
def test_production_api_trio_on_gpu_session_reuse_and_graph_replay(trio,tmp_path,profile,device):
    result,paths=gpu_linked_trio(trio,tmp_path,profile,device)
    tensor=dict(allocation_bytes=64,extents=(8,),bit_strides=(64,),dtype_id=DTYPES['u64']['id'],
                alignment=16)
    with CudaBackend(tmp_path/'pipeline.cubin',result.kernel_symbol,dynamic_smem_bytes=result.dyn_smem_bytes,
                     cluster=result.cluster,block=(result.threads,1,1)) as backend, Fixture(backend) as fixture:
        fixture.tensor('input',data=struct.pack('<8Q',*range(8)),**tensor)
        fixture.tensor('output',data=None,output=True,**tensor)
        stream=backend.stream_token
        def replay(launch):
            launch(stream)
            backend.synchronize()
            return struct.unpack('<8Q',fixture.readback()['output'])
        with broker.open_host(result,paths,owner_args({'user':1}),fixture.resources,{'M':96},backend) as session:
            assert session.geometry['grid']==(3,1,1)
            assert [replay(session.launch) for _ in range(3)]==[OUTPUT]*3
            # Capturing the session's launch proves nothing forbidden sits on its launch path.
            with backend.capture(stream,lambda: session.launch(stream)) as graph:
                assert [replay(graph.launch) for _ in range(3)]==[OUTPUT]*3

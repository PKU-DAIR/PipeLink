#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"

# The packed inventory is the sole source of the provider's ABI identity.
uv run --project "$ROOT/pipec" --locked python - "$HERE" "$ROOT" "${1:-$HERE/../port/build/artifacts/interpreter.plink}" <<'PY'
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from pipec2.container import read
from pipec2.host.abi import ABI_TYPES
from pipec2.host.provider import build_provider

here, root, artifact = map(Path, sys.argv[1:])
cuda = Path(os.environ['PIPELINK_CUDA_HOME'])
mk = root / 'third_party/Megakernels'
manifest = read(artifact)[0]
host = manifest['host']
provider = host['provider']
inventory = provider['abi_types']
build = here / 'build'
build.mkdir(exist_ok=True)
includes = [here, root / 'pipec/src/pipec2/host/include',
            mk / 'ThunderKittens/include', mk / 'include', mk / 'demos/low-latency-llama']
flags = ['-x', 'cu', '-std=c++20', '-O2', '-DKITTENS_HOPPER', '-DKITTENS_BLACKWELL',
         '--expt-extended-lambda', '--expt-relaxed-constexpr', '-arch=sm_103a',
         *['-I' + str(path) for path in includes], '-L' + str(cuda / 'lib64/stubs'), '-lcuda']
with tempfile.TemporaryDirectory(prefix='mk-provider-abi-') as temporary:
    header = Path(temporary) / 'owner_abi.cuh'
    header.write_text('#include "mk_host_abi.h"\n#include "llama.cuh"\n'
                      'using mk_globals = ::llama_1b_globals;\nusing namespace mk_host;\n')
    observed = build_provider(
        [here / 'provider.cu'], build / 'provider.so', owner='interpreter',
        owner_contract_digest=provider['owner_contract_digest'],
        profile=manifest['pipelink_abi']['target']['profile'],
        dtypes_digest=host['target']['dtypes'],
        type_ids={row['name']: row['id'] for row in inventory if row['name'] not in ABI_TYPES},
        extra_headers=[header],
        extra_types={row['name']: row['fields'] for row in inventory if row['name'] not in ABI_TYPES},
        compiler=str(cuda / 'bin/nvcc'), compile_flags=flags)
    if observed != inventory:
        raise RuntimeError('provider ABI inventory differs from packed interpreter')
subprocess.run([str(cuda / 'bin/nvcc'), '-std=c++17', '-O2', '-DMK_PROVIDER_SELFTEST',
                '-I' + str(here), '-I' + str(root / 'pipec/src/pipec2/host/include'),
                str(here / 'provider.cu'), '-L' + str(cuda / 'lib64/stubs'), '-lcuda', '-ldl',
                '-o', str(build / 'provider_selftest')], check=True)
print(build / 'provider.so')
print('GPU selftest (explicit opt-in):', build / 'provider_selftest', build / 'provider.so')
PY

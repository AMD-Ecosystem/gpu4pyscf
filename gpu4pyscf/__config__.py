# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import cupy

num_devices = cupy.cuda.runtime.getDeviceCount()

props = cupy.cuda.runtime.getDeviceProperties(0)

_is_hip = bool(getattr(cupy.cuda.runtime, 'is_hip', False))
GB = 1024*1024*1024
min_ao_blksize = 256        # maxisum batch size of AOs
min_grid_blksize = 64*64    # maximum batch size of grids for DFT
ao_aligned = 32             # global AO alignment for slicing
grid_aligned = 256          # 256 alignment for grids globally

# Use smaller blksize for old gaming GPUs
if props['totalGlobalMem'] < 16 * GB:
    min_ao_blksize = 64
    min_grid_blksize = 64*64

# Use 90% of the global memory for CuPy memory pool
mem_fraction = 0.9
cupy.get_default_memory_pool().set_limit(fraction=mem_fraction)

# sharedMemPerBlockOptin is a CUDA-only device attribute; CuPy-ROCm does not
# expose it. Fall back to sharedMemPerBlock (the per-block LDS limit, 64 KB on
# gfx90a) when it is absent so the opt-in branch is skipped on ROCm. On CUDA
# the key is present and behaviour is unchanged.
if props.get('sharedMemPerBlockOptin', 0) > 65536:
    shm_size = props['sharedMemPerBlockOptin']
    # Resever space for system use
    shm_size -= 2048
else:
    shm_size = props['sharedMemPerBlock']

# Check P2P data transfer is available
_p2p_access = True
if num_devices > 1:
    for src in range(num_devices):
        for dst in range(num_devices):
            if src != dst:
                can_access_peer = cupy.cuda.runtime.deviceCanAccessPeer(src, dst)
                _p2p_access &= can_access_peer

def _probe_smid_pool_slots(mp_count):
    '''Number of scratch-pool slots a get_smid()-indexed HIP kernel can address.

    On AMD GPUs get_smid() (HIP __smid(), the hardware CU-slot id) can exceed
    multiProcessorCount because the (SE,CU) id packing is sparse: on gfx90a it
    reaches 125 while multiProcessorCount is 104. A scratch pool indexed by
    get_smid()*POOL_SIZE must therefore be sized to the smid range, not the
    active-CU count, or resident blocks on high-id CUs write past the pool
    (silent device-heap corruption). Probe the actual range once via libgvhf_rys
    and round up with margin so the slot count provably covers the full id range.
    '''
    from gpu4pyscf.lib.utils import load_library
    libvhf_rys = load_library('libgvhf_rys')
    libvhf_rys.probe_smid_range.restype = ctypes.c_int
    libvhf_rys.probe_smid_range.argtypes = [ctypes.c_int]
    observed = libvhf_rys.probe_smid_range(ctypes.c_int(mp_count))
    if observed <= 0:
        # Probe failed; fall back to a safe gfx9 ceiling (smid encodes
        # SE_ID[5:4]<<4 | CU_ID[3:0], so 0..127, i.e. 128 slots) widened to
        # cover larger CU/SE layouts on other AMD architectures.
        return max(256, mp_count)
    lower = max(observed, mp_count)
    slots = 1
    while slots < lower:
        slots <<= 1
    return slots

if _is_hip:
    pool_slots = _probe_smid_pool_slots(props['multiProcessorCount'])
else:
    pool_slots = props['multiProcessorCount']

# Overwrite the above settings using the global pyscf configs
from pyscf.__config__ import * # noqa

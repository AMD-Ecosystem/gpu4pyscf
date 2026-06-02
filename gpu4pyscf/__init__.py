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

__version__ = '1.7.1'

from . import _patch_pyscf

# ROCm/HIP port -- Milestone 1 (HF / integrals). The DFT-dependent subpackages
# (grad, hessian, solvent, dft, tdscf, nac) transitively load libgdft and the
# CUTLASS grouped-GEMM kernels, neither of which is part of the HF / integral
# milestone (gen_grids needs a HIP double3 rewrite; the XC functional path
# needs libxc-HIP; grouped GEMM needs a hipBLAS/CK reimplementation). Importing
# them here would fail at package import on ROCm, so under HIP only lib and scf
# (the HF J/K path) are wired eagerly; the rest stay importable on demand once
# their kernels are ported. On CUDA every subpackage is imported as before.
import cupy as _cupy
if getattr(_cupy.cuda.runtime, 'is_hip', False):
    from . import lib, scf
else:
    from . import lib, grad, hessian, solvent, scf, dft, tdscf, nac
del _cupy

# Overwrite the cupy memory allocator. Make memory pool manage small-sized
# arrays only.
lib.cupy_helper.set_conditional_mempool_malloc()

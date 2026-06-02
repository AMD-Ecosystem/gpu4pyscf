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

# ROCm/HIP port -- Milestone 1. The molecular HF path reaches this subpackage
# only for the libpbc handle (pbc.df.ft_ao) and the int3c2e task helpers; the
# periodic density-fitting drivers (fft / aft / df) belong to a later milestone
# and their eager import here forms a cycle (aft -> pp_int -> pbc.gto.int1e)
# that the molecular import order would otherwise trip. They stay importable on
# demand. On CUDA every driver is wired eagerly as before.
import cupy as _cupy
if not getattr(_cupy.cuda.runtime, 'is_hip', False):
    from . import fft
    from . import aft
    from . import df
    from .fft import FFTDF
    from .df import GDF
    from .aft import AFTDF
del _cupy

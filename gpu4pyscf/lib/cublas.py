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
import ctypes.util
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas #NOQA

# Under ROCm the cuBLAS backend is hipBLAS, so the dense-vendor library is
# libhipblas.so rather than libcublas.so. Load it lazily and tolerate its
# absence: this module currently exposes no extra entry points (the handle
# below is the only consumer and the cupy_backends.cuda.libs.cublas backend,
# already ROCm-aware, is the real BLAS path).
def _load_blas_library():
    for name in ('cublas', 'hipblas'):
        path = ctypes.util.find_library(name)
        if path is not None:
            return ctypes.CDLL(path)
    return None

libcublas = _load_blas_library()

# This needs to be moved into functions, and lazy evaluate
_handle = device.get_cublas_handle() #NOQA

# NOTE: add modified cublas function here

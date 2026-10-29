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


import numpy as np
import cupy
import ctypes
from cupy_backends.cuda.libs import cusolver
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device

from gpu4pyscf.lib.utils import load_library

from ctypes.util import find_library

# ROCm/HIP build: cuSOLVER's dense generalized eigensolver maps to hipSOLVER.
# The hipsolverDn{D,Z}sygvd/hegvd entry points are ABI-identical to the
# cusolverDn ones (same argument order and types, confirmed against
# hipsolver-dense.h), so only three things differ from the CUDA path: the
# library name (libhipsolver.so), the enum *values* (hipSOLVER renumbers
# itype/jobz/uplo away from cuSOLVER's 1/1/0), and the handle -- CuPy-ROCm's
# device.get_cusolver_handle() returns a rocSOLVER/rocBLAS handle, not a
# hipSOLVER handle, so a hipSOLVER handle is created here and bound to the
# current stream. The cholesky() path below goes through CuPy's own potrf
# backend (already ROCm) and needs no change.
_IS_ROCM = getattr(cupy.cuda.runtime, 'is_hip', False)

if _IS_ROCM:
    libcusolver = ctypes.CDLL('libhipsolver.so')
    CUSOLVER_EIG_TYPE_1 = 211
    CUSOLVER_EIG_TYPE_2 = 212
    CUSOLVER_EIG_TYPE_3 = 213
    CUSOLVER_EIG_MODE_NOVECTOR = 201
    CUSOLVER_EIG_MODE_VECTOR = 202
    _FILL_MODE_LOWER = 122  # HIPBLAS_FILL_MODE_LOWER
    libcusolver.cusolverDnDsygvd_bufferSize = libcusolver.hipsolverDnDsygvd_bufferSize
    libcusolver.cusolverDnDsygvd = libcusolver.hipsolverDnDsygvd
    libcusolver.cusolverDnZhegvd_bufferSize = libcusolver.hipsolverDnZhegvd_bufferSize
    libcusolver.cusolverDnZhegvd = libcusolver.hipsolverDnZhegvd

    libcusolver.hipsolverCreate.restype = int
    libcusolver.hipsolverCreate.argtypes = [ctypes.c_void_p]
    libcusolver.hipsolverSetStream.restype = int
    libcusolver.hipsolverSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _hipsolver_handle = ctypes.c_void_p()
    if libcusolver.hipsolverCreate(ctypes.byref(_hipsolver_handle)) != 0:
        raise RuntimeError('hipsolverCreate failed')
else:
    libcusolver = find_library('cusolver')
    if libcusolver is not None:
        libcusolver = ctypes.CDLL(libcusolver)
    else:
        libcusolver = load_library('libcusolver', path='nvidia/cusolver/lib')
    CUSOLVER_EIG_TYPE_1 = 1
    CUSOLVER_EIG_TYPE_2 = 2
    CUSOLVER_EIG_TYPE_3 = 3
    CUSOLVER_EIG_MODE_NOVECTOR = 0
    CUSOLVER_EIG_MODE_VECTOR = 1
    _FILL_MODE_LOWER = cublas.CUBLAS_FILL_MODE_LOWER

# workspace size (lwork) provided by the cusolver*_bufferSize is an 32-bit
# integer. For arrays above this dimension, the workspace size would overflow.
MAX_EIGH_DIM = 23150

libcusolver.cusolverDnDsygvd_bufferSize.restype = int
libcusolver.cusolverDnDsygvd.restype = int

_buffersize = {}

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnDsygvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnDsygvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnZhegvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnZhegvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

def eigh(h, s, overwrite=False):
    '''
    solve generalized eigenvalue problem
    '''
    assert h.dtype == s.dtype
    assert h.dtype in (np.float64, np.complex128)
    n = h.shape[0]
    if h.dtype == np.complex128 and h.flags.c_contiguous:
        # zhegvd requires the matrices in F-order. For hermitian matrices,
        # .T.copy() is equivalent to .conj()
        A = h.conj()
        B = s.conj()
    elif overwrite:
        A = h
        B = s
    else:
        A = h.copy()
        B = s.copy()
    if _IS_ROCM:
        _handle = _hipsolver_handle
        libcusolver.hipsolverSetStream(
            _handle, ctypes.c_void_p(cupy.cuda.get_current_stream().ptr))
    else:
        _handle = device.get_cusolver_handle()
    w = cupy.zeros(n)

    # TODO: reuse workspace
    if (h.dtype, n) in _buffersize:
        lwork = _buffersize[h.dtype, n]
    else:
        lwork = ctypes.c_int(0)
        if h.dtype == np.float64:
            fn = libcusolver.cusolverDnDsygvd_bufferSize
        else:
            fn = libcusolver.cusolverDnZhegvd_bufferSize
        status = fn(
            _handle,
            CUSOLVER_EIG_TYPE_1,
            CUSOLVER_EIG_MODE_VECTOR,
            _FILL_MODE_LOWER,
            n,
            A.data.ptr,
            n,
            B.data.ptr,
            n,
            w.data.ptr,
            ctypes.byref(lwork)
        )
        lwork = lwork.value
        _buffersize[h.dtype, n] = lwork

        if status != 0:
            raise LinAlgError("failed in buffer size")

    if h.dtype == np.float64:
        fn = libcusolver.cusolverDnDsygvd
    else:
        fn = libcusolver.cusolverDnZhegvd
    work = cupy.empty(lwork, dtype=h.dtype)
    devInfo = cupy.empty(1, dtype=np.int32)
    status = fn(
        _handle,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        _FILL_MODE_LOWER,
        n,
        A.data.ptr,
        n,
        B.data.ptr,
        n,
        w.data.ptr,
        work.data.ptr,
        lwork,
        devInfo.data.ptr,
    )

    if status != 0:
        raise LinAlgError("failed in eigh kernel")
    return w, A.T

def cholesky(A):
    n = len(A)
    if A.flags.f_contiguous:
        A = A.T
    x = A.copy(order='C')
    handle = device.get_cusolver_handle()
    if A.dtype == np.float64:
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    else:
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize
    buffersize = potrf_bufferSize(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=A.dtype)
    dev_info = cupy.empty(1, dtype=np.int32)
    potrf(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)

    if dev_info[0] != 0:
        raise LinAlgError('failed to perform Cholesky Decomposition')
    cupy.linalg._util._tril(x,k=0)
    return x

class LinAlgError(RuntimeError):
    pass

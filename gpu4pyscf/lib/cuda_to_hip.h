/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Single CUDA-to-HIP compatibility shim for the ROCm/HIP port.
 * Force-included on every HIP translation unit via -include; on a CUDA build
 * this file is never reached (CMake force-includes it only under USE_HIP).
 * The kernel sources keep their CUDA spelling; this header is the one place
 * that knows about HIP.
 */

#ifndef GPU4PYSCF_CUDA_TO_HIP_H
#define GPU4PYSCF_CUDA_TO_HIP_H

#include <hip/hip_runtime.h>

// gpu4pyscf only reports this value (scf/jk.py reads it for diagnostics, no
// version-gated code path), so surface the HIP runtime version under ROCm.
#ifndef CUDA_VERSION
#define CUDA_VERSION (HIP_VERSION_MAJOR * 1000 + HIP_VERSION_MINOR * 10)
#endif

typedef hipError_t              cudaError_t;
typedef hipStream_t             cudaStream_t;
typedef hipDeviceProp_t         cudaDeviceProp;
typedef hipFuncAttributes       cudaFuncAttributes;
typedef hipMemcpyKind           cudaMemcpyKind;

#define cudaSuccess             hipSuccess
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault       hipMemcpyDefault

#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemcpyAsync         hipMemcpyAsync
#define cudaMemcpyToSymbol      hipMemcpyToSymbol
#define cudaMemset              hipMemset
#define cudaMemsetAsync         hipMemsetAsync
#define cudaMemcpy2DAsync       hipMemcpy2DAsync
#define cudaMemGetInfo          hipMemGetInfo
#define cudaGetSymbolAddress    hipGetSymbolAddress

// CUDA's cudaHostGetDevicePointer is templated on the device-pointer type;
// hip's takes void**. Wrap to accept a typed pointer-to-pointer.
template <typename T> static inline
hipError_t _g4p_host_get_device_pointer(T **dev, void *host, unsigned int flags)
{
    return hipHostGetDevicePointer((void **)dev, host, flags);
}
#define cudaHostGetDevicePointer(dev, host, flags) \
        _g4p_host_get_device_pointer((dev), (host), (flags))

#define cudaGetDevice           hipGetDevice
#define cudaSetDevice           hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceReset         hipDeviceReset
#define cudaDeviceSynchronize   hipDeviceSynchronize

#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString
#define cudaGetErrorName        hipGetErrorName

#define cudaFuncAttributeMaxDynamicSharedMemorySize \
        hipFuncAttributeMaxDynamicSharedMemorySize

// As with cudaFuncSetAttribute, the CUDA cudaFuncGetAttributes is templated on
// the kernel type; wrap the hip const void* form to accept the kernel symbol.
template <typename F> static inline
hipError_t _g4p_func_get_attributes(hipFuncAttributes *attr, F func)
{
    return hipFuncGetAttributes(attr, (const void *)func);
}
#define cudaFuncGetAttributes(attr, func) \
        _g4p_func_get_attributes((attr), (func))

/*
 * CUDA's cudaFuncSetAttribute is templated on the kernel type, so call sites
 * pass the kernel symbol directly. hipFuncSetAttribute takes a const void*,
 * so wrap it to accept the kernel reference and cast (matching CUDA's ABI).
 */
template <typename F> static inline
hipError_t _g4p_func_set_attribute(F func, hipFuncAttribute attr, int value)
{
    return hipFuncSetAttribute((const void *)func, attr, value);
}
#define cudaFuncSetAttribute(func, attr, value) \
        _g4p_func_set_attribute((func), (attr), (value))

#define cudaStreamDefault       hipStreamDefault
#define cudaStreamNonBlocking   hipStreamNonBlocking
#define cudaStreamCreate        hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy       hipStreamDestroy
#define cudaStreamSynchronize   hipStreamSynchronize

/*
 * The wave64 subgroup-reduction fix lives in cuda_to_hip_shfl.h, force-included
 * only on the gvhf-md J-engine translation units (see gvhf-md/CMakeLists.txt).
 * It is intentionally NOT defined here: this header is force-included on every
 * HIP TU, and a global __shfl_down_sync macro collides with HIP's own
 * mask-aware __shfl_down_sync overloads pulled in by hipCUB / the bf16 headers
 * (used by the pbc int3c2e kernels).
 */

#endif // GPU4PYSCF_CUDA_TO_HIP_H

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
 * wave64 subgroup-reduction fix for the gvhf-md J-engine (the central
 * correctness change of the ROCm/HIP port) and the multigrid_v3 kernels.
 * Force-included ONLY on the gvhf-md and multigrid_v3 translation units that
 * keep the CUDA __shfl_down_sync spelling;
 * it is deliberately not in the global compat header because a global
 * __shfl_down_sync macro collides with HIP's own mask-aware overloads that
 * hipCUB / the bf16 headers (used elsewhere, e.g. the pbc int3c2e kernels)
 * bring in.
 *
 * The gvhf-md kernels reduce partial Coulomb contributions over a logical
 * subgroup of W consecutive lanes (W is 8, 16 or 32 depending on the kernel)
 * with the CUDA idiom
 *
 *     mask = ((1u << W) - 1) << (group_id * W);   // a W-bit lane mask
 *     for (offset = W/2; offset; offset /= 2)
 *         val += __shfl_down_sync(mask, val, offset);
 *
 * Two things break on gfx90a with the CUDA spelling: (a) ROCm's
 * __shfl_down_sync requires a 64-bit mask, so the 32-bit literal will not
 * compile; (b) a 64-lane wavefront holds 64/W subgroups, but a CUDA-shaped
 * 32-bit mask only describes the subgroups living in the low 32 lanes, so
 * lanes 32-63 would be mis-partitioned.
 *
 * HIP's width-parameterized __shfl_down(val, offset, width) partitions the
 * whole wavefront (any wave size) into width-lane segments and stays inside
 * the calling lane's segment. The subgroup width W is exactly popcount(mask),
 * recovered uniformly from every call site, so the same shim is correct for
 * the 8-, 16- and 32-lane reductions on both wave32 (gfx10/11, 2 subgroups of
 * 16 in a 32-lane wave) and wave64 (gfx9, 4 subgroups of 16 in a 64-lane wave)
 * with no per-call-site edit. Lanes the CUDA mask never named (32-63 on
 * wave64) now reduce within their own width-W segment, which is what the
 * row-local reduction intends. Because the thread block lays threads out as
 * sq_id = tx + 16*ty (blockDim.x = 16), thread sq_id maps to hardware lane
 * sq_id, so a width-16 segment is exactly one ty-row -- matching the CUDA
 * semantics on both wave sizes.
 */
#ifndef GPU4PYSCF_CUDA_TO_HIP_SHFL_H
#define GPU4PYSCF_CUDA_TO_HIP_SHFL_H

template <typename T> __device__ __forceinline__
T _g4p_shfl_down_sub(unsigned long long mask, T val, int offset)
{
    return __shfl_down(val, offset, __popcll(mask));
}
#define __shfl_down_sync(mask, val, offset) \
        _g4p_shfl_down_sub((unsigned long long)(mask), (val), (offset))

#endif // GPU4PYSCF_CUDA_TO_HIP_SHFL_H

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

if(NOT USE_HIP)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")
endif()

# libxc.cu here is gpu4pyscf's own XC-contraction kernel TU (the GDFT_xc_*
# entry points dft/libxc.py binds), not the external libxc functional library
# (the deferred BUILD_LIBXC ExternalProject). These are grid / AO-value /
# density-contraction kernels with block-level reductions only (no warp
# shuffles), so they are wave-size agnostic. libgdft is built under HIP because
# the molecular HF J/K path import-cascades through gpu4pyscf.dft.libxc, which
# load_library('libgdft') at import; the DFT functional evaluation path itself
# stays deferred (no external libxc).
add_library(gdft SHARED
  nr_eval_gto.cu
  contract_rho.cu
  gen_grids.cu
  nr_numint_sparse.cu
  vv10.cu
  libxc.cu
)

g4p_set_hip_language(gdft)
set_target_properties(gdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
if(USE_HIP)
  set_target_properties(gdft PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

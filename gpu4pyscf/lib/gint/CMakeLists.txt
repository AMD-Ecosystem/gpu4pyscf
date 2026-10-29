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
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
add_library(gint SHARED 
  g2e.c
  pair_data.c
  bpcache.cu
  constant.cu
  #nr_fill_ao_ints.cu
  nr_fill_ao_int3c1e.cu
  nr_fill_ao_int3c1e_ip.cu
  nr_fill_ao_int3c1e_ipip.cu
  nr_fill_ao_int3c2e.cu
  #nr_fill_ao_int3c2e_ip1.cu
  #nr_fill_ao_int3c2e_ip2.cu
  #nr_fill_ao_int3c2e_ipip1.cu
  #nr_fill_ao_int3c2e_ipip2.cu
  #nr_fill_ao_int3c2e_ip1ip2.cu
  #nr_fill_ao_int3c2e_ipvip1.cu
  j_engine_matrix_reorder.c
  rys_roots_dat.cu
  cart2sph.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

g4p_set_hip_language(gint)
if(USE_HIP)
  # The legacy int2e fill kernels (g2e_root_n.cu, included by nr_fill_ao_ints.cu)
  # instantiate huge GOUTSIZE local arrays for the high angular-momentum cases
  # (NROOTS 7/8); at -O2 device codegen spills > 360 KB to scratch and
  # amdgcn-link rejects the frame. Final device codegen happens at the gint
  # relocatable-device-code link, so the opt level must be lowered there too.
  # gint is the legacy ERI engine (the hot path is gvhf-rys), so -O1 here is fine.
  set_source_files_properties(nr_fill_ao_ints.cu PROPERTIES COMPILE_OPTIONS "-O1")
endif()
set_target_properties(gint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
  HIP_SEPARABLE_COMPILATION ON)
if(USE_HIP)
  target_link_options(gint PRIVATE -O1)
endif()

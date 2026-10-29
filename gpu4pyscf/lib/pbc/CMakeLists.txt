if(NOT USE_HIP)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")
endif()

# libpbc supplies the 1e-integral kernels (PBCint1e_kin/ovlp, PBCsort_pair_ij,
# PBCovlp_mask_estimation) that the molecular HF hcore and scf/jk.py Schwarz
# cache call at this pinned base, so it is part of the HF milestone. The shared
# warp-max reduction in int3c2e_create_tasks.cuh is made wave-size correct
# (see that header). The wave64-broken __ballot path lives only in the
# uncompiled int3c2e_create_tasks_o1.cuh and is deferred with the periodic
# int3c2e milestone.
add_library(pbc SHARED
  pbc_driver.cu
  ft_ao.cu ft_ao_ip1.cu contract_ft_ao.cu
  fill_int3c2e.cu contract_int3c2e.cu ejk_int3c2e_ip1.cu
  fill_int2c2e.cu fill_int2c2e_ip1.cu e_int2c2e_ip1.cu
  contract_int3c2e_pvp.cu
  overlap.cu ppnl.cu
  supmol_sr_estimator.cu
  rys_roots_dat.cu
  rys_contract_k.cu rys_contract_jk_ip1.cu
  rys_contract_j.cu
  unrolled_rys_k.cu
  nr_eval_gto.cu
  sorting.c write_cderi.c int3c2e_recontract.c
  screen.cu
  fill_triu.cu
)

g4p_set_hip_language(pbc)
set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
  HIP_SEPARABLE_COMPILATION ON)
if(USE_HIP)
  set_target_properties(pbc PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

target_link_libraries(pbc OpenMP::OpenMP_C)

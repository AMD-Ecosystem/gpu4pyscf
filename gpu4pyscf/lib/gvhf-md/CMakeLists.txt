if(NOT USE_HIP)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")
endif()

add_library(gvhf_md SHARED
  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu
  md_indices.cu md_j_driver.cu md_pairdata.c 
  contract_int3c2e.cu
  pbc_md_contract_j.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

g4p_set_hip_language(gvhf_md)
if(USE_HIP)
  # The wave64 subgroup-reduction shim is force-included only on the three
  # J-engine TUs that keep the CUDA __shfl_down_sync(mask, ...) spelling, so
  # the global compat header stays free of a __shfl_down_sync macro that would
  # collide with HIP's own mask-aware overloads elsewhere in the tree.
  set_source_files_properties(
    md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu
    PROPERTIES COMPILE_OPTIONS "-include;${PROJECT_SOURCE_DIR}/cuda_to_hip_shfl.h")
endif()
set_target_properties(gvhf_md PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
  HIP_SEPARABLE_COMPILATION ON)

target_link_libraries(gvhf_md OpenMP::OpenMP_C)

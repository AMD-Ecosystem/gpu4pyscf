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

cmake_minimum_required (VERSION 3.19 FATAL_ERROR) # 3.19 is required by cutlass

# ROCm/HIP port. Under USE_HIP the .cu sources are retagged
# LANGUAGE HIP and compiled by the HIP toolchain; the single compat header
# gpu4pyscf/lib/cuda_to_hip.h (force-included below) maps the CUDA spellings
# to HIP so the kernel sources stay in CUDA form. The CUDA build is unchanged.
option(USE_HIP "Build the CUDA kernel libraries for ROCm/HIP" OFF)
if(USE_HIP)
  project (gpu4pyscf C CXX HIP)
  if(DEFINED CMAKE_HIP_ARCHITECTURES)
    # honour the cache value (e.g. gfx90a / gfx1100 / gfx1151)
  elseif(DEFINED ENV{HIP_ARCHITECTURES})
    set(CMAKE_HIP_ARCHITECTURES "$ENV{HIP_ARCHITECTURES}")
  endif()
  message("HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
else()
  project (gpu4pyscf C CXX CUDA Fortran)
endif()

set(CMAKE_C_STANDARD "99")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_HIP)
  # Force-include the compat shim on every HIP TU so the cuda* spellings and
  # the wave64 subgroup-reduction fix are defined before any source is parsed.
  # Pin fp-contract so the J/K fingerprints match the CPU PySCF reference to
  # the test's 7 d.p. bar (clang-HIP defaults to -ffp-contract=fast).
  # -fvisibility=default: the integral kernels share `extern __device__` Rys-root
  # tables and `__constant__` index tables across translation units; with HIP RDC
  # device globals default to protected visibility and amdgcn-link cannot resolve
  # them across objects, so force default visibility for the device link.
  # -fgpu-rdc: the integral libraries split device code across translation units
  # (the CUDA build uses CUDA_SEPARABLE_COMPILATION). CMake's HIP_SEPARABLE_COMPILATION
  # does not inject the relocatable-device-code flag, so set it explicitly here and
  # add --hip-link -fgpu-rdc on the shared-library link below.
  # -Wno-register: a few unrolled kernels still use the (C++17-removed) register
  # keyword; nvcc tolerates it, clang-HIP rejects it, so downgrade to a warning.
  string(APPEND CMAKE_HIP_FLAGS
    " -DUSE_HIP -include ${PROJECT_SOURCE_DIR}/cuda_to_hip.h -ffp-contract=on -fvisibility=default -fgpu-rdc -Wno-register")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fgpu-rdc --hip-link")
  # Resolve `#include <cuda_runtime.h>` in the kernel sources to the compat shim.
  include_directories(BEFORE ${PROJECT_SOURCE_DIR}/hip_compat)
endif()

# For better performance on A100, the option
# -DCUDA_ARCHITECTURES="70-real;80-real" can be added to cmake command
if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
elseif(NOT USE_HIP)
  set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
endif()
message("CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
message("Build type ${CMAKE_BUILD_TYPE}")

set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O2")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -fno-math-errno -fno-strict-overflow")
    endif()
  endif()
endif()

option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

#if(BLKSIZE)
#  message("Block size for integral buffer ${BLKSIZE}")
#  add_definitions(-DBLKSIZE=${BLKSIZE})
#endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib)

#list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

# A .cu retagged LANGUAGE HIP is compiled by the HIP toolchain under USE_HIP.
function(g4p_set_hip_language target)
  if(USE_HIP)
    get_target_property(_g4p_srcs ${target} SOURCES)
    foreach(_g4p_src ${_g4p_srcs})
      if(_g4p_src MATCHES "\\.cu$")
        set_source_files_properties(${_g4p_src} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()
endfunction()

if(USE_HIP)
  option(BUILD_CUTLASS "Using cutlass" OFF)
else()
  option(BUILD_CUTLASS "Using cutlass" ON)
endif()
if(BUILD_CUTLASS)
  include(ExternalProject)
  ExternalProject_Add(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v3.4.0
    CONFIGURE_COMMAND ""  # No configure step
    BUILD_COMMAND ""      # No build step
    INSTALL_COMMAND ""    # No install step
    TEST_COMMAND ""       # No test step
  )

  # ExternalProject_Add automatically populates this variable
  ExternalProject_Get_Property(cutlass SOURCE_DIR)
  set(cutlass_SOURCE_DIR ${SOURCE_DIR})
endif()

add_subdirectory(gint)
add_subdirectory(gvhf)
add_subdirectory(cupy_helper)
add_subdirectory(gvhf-rys)
add_subdirectory(gvhf-md)
# The pinned base (#773, "Optimize J/K build for molecular code") routes the
# molecular HF hcore and the scf/jk.py Schwarz-condition cache through the pbc
# 1e-integral kernels (libpbc) and import-cascades through gpu4pyscf.dft.libxc
# (libgdft), so even plain RHF get_jk needs libgecp, libgdft and libpbc in
# addition to the J/K engine libraries. All three build under HIP; gdft and pbc
# kernels are made wave-size correct alongside the gvhf-md J-engine fix.
add_subdirectory(ecp)
add_subdirectory(gdft)
add_subdirectory(pbc)

# Milestone 1 of the ROCm/HIP port builds the integral / HF (J/K) libraries
# plus the import-/runtime-coupled libgecp, libgdft and libpbc. The grouped-GEMM
# (cupy_helper CUTLASS), external libxc functional library, DFT functional
# evaluation, solvent, sem and multigrid are deferred to later milestones.
if(NOT USE_HIP)
  add_subdirectory(sem)

  option(BUILD_SOLVENT "Using SMD solvent" ON)
  if(BUILD_SOLVENT)
    add_subdirectory(solvent)
  endif()

  add_subdirectory(multigrid)
endif()

if(USE_HIP)
  option(BUILD_LIBXC "Using libxc for DFT" OFF)
else()
  option(BUILD_LIBXC "Using libxc for DFT" ON)
endif()
if(BUILD_LIBXC)
  include(ExternalProject)
  ExternalProject_Add(libxc
    GIT_REPOSITORY https://github.com/wxj6000/libxc.git
    GIT_TAG b225c254c063e1de835a4425115c9a6377478b32
    PATCH_COMMAND git apply --reject ${PROJECT_SOURCE_DIR}/libxc_prune.patch || true
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DENABLE_FORTRAN=OFF -DDISABLE_KXC=OFF -DDISABLE_LXC=ON -DDISABLE_FHC=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
            -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
    CMAKE_CACHE_ARGS -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
  )
endif()

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
 * HIP-build shim: the pbc int3c2e kernels include <cub/cub.cuh> and use
 * cub::BlockScan / cub::BlockReduce. hipCUB provides the same block-level
 * primitives over rocPRIM, so resolve the CUB include to hipCUB and alias the
 * cub namespace. Only the block primitives the kernels touch are exercised.
 */
// <complex.h> (included ahead of this header by multigrid_v2/screen.cu)
// defines the macro I, which rocPRIM uses as a template parameter name.
#ifdef I
#undef I
#endif
#include <hipcub/hipcub.hpp>

namespace cub = hipcub;

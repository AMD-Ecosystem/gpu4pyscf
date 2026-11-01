/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
 * Variants of type2_cart_ip1_general, type2_cart_ipipv and type2_cart_ipvip
 * for GPUs whose shared memory per block (64 KB on AMD GPUs) cannot hold the
 * scratch of type2_cart_kernel for g shells. The angular factors omegai and
 * omegaj, the largest arrays, are kept in a per-block global-memory
 * workspace; the radial and angular tables read in the inner loop stay in
 * shared memory. The grid is sized to the resident blocks and each block
 * loops over tasks, so the workspace does not grow with the task count.
 */

__host__ __device__ inline
int type2_omega_size(const int LI, const int LC){
    return (LI+LC+2)/2 * (LI+1)*(LI+2)*(LI+3)/6 * (2*LC+1);
}

__host__ __device__ inline
int type2_lowmem_smem_size(const int LI, const int LJ, const int LC){
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    return (LI+LJ+1)*(LI+LC+1)*(LJ+LC+1)  // rad_all
         + (LI+1)*nfi*(LI+LC+1)           // angi
         + (LJ+1)*nfj*(LJ+LC+1);          // angj
}

__host__ __device__ inline
int type2_lowmem_work_size(const int LI, const int LJ, const int LC){
    return type2_omega_size(LI, LC) + type2_omega_size(LJ, LC);
}

// type2_cart_kernel with omegai and omegaj in the global workspace omega
template <int orderi, int orderj> __device__
void type2_cart_kernel_lowmem(double *gctr,
                const int LI, const int LJ, const int LC,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env,
                double *omega)
{
    extern __shared__ double smem[];

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    double* omegai = omega;
    double* omegaj = omegai + type2_omega_size(LI, LC);

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    double* rad_all = smem;
    set_shared_memory(rad_all, (LI+LJ+1)*(LI+LC+1)*(LJ+LC+1));

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    double radi[AO_LMAX+ECP_LMAX+orderi+1];
    type2_facs_rad<orderi>(radi, LI+LC, npi, dca, ci, ai);
    double radj[AO_LMAX+ECP_LMAX+orderj+1];
    type2_facs_rad<orderj>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }
    double ur_tmp = ur;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*(LI+LC+1)*(LJ+LC+1);
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur_tmp, prad+i*(LJ+LC+1)+j);
        }}
        const int ir = threadIdx.x;
        ur_tmp *= r128[ir];
    }
    __syncthreads();

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = rad_all + (LI+LJ+1)*(LI+LC+1)*(LJ+LC+1);
    double* angj = angi + (LI+1)*nfi*(LI+LC+1);

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < 2*LC+1; m++){
        type2_ang(angi, LI, LC, rca, omegai+m);
        type2_ang(angj, LJ, LC, rcb, omegaj+m);
        __syncthreads();
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                const int LIC1 = LI+LC+1;
                const int LJC1 = LJ+LC+1;
                double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                double *prad = rad_all + (k+l)*LIC1*LJC1;

                double reg_angi[AO_LMAX+ECP_LMAX+orderi+1];
                double reg_angj[AO_LMAX+ECP_LMAX+orderj+1];
                for (int p = 0; p < LIC1; p++){reg_angi[p] = pangi[p];}
                for (int q = 0; q < LJC1; q++){reg_angj[q] = pangj[q];}
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * reg_angi[p] * reg_angj[q];
                }}
            }}
            gctr[ij] += fac*s;
        }
        __syncthreads();
    }
}

__global__
void type2_cart_ip1_lowmem(double *gctr,
                const int LI, const int LJ, const int LC,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env,
                double *work, const int work_size)
{
    double *omega = work + (size_t)blockIdx.x * work_size;
    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    constexpr int NFI_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int NFJ_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf[NFI_MAX*NFJ_MAX];

    for (int task_id = blockIdx.x; task_id < ntasks; task_id += gridDim.x){
        const int ish = tasks[task_id];
        const int jsh = tasks[task_id + ntasks];
        const int ksh = tasks[task_id + 2*ntasks];
        const int ioff = ao_loc[ish];
        const int joff = ao_loc[jsh];
        const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
        double *pgctr = gctr + 3*ecp_id*nao*nao + ioff*nao + joff;

        for (int ij = threadIdx.x; ij < NF_MAX*NF_MAX*3; ij+=blockDim.x){
            gctr_smem[ij] = 0.0;
        }
        __syncthreads();

        type2_cart_kernel_lowmem<1,0>(
            buf, LI+1, LJ, LC,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env, omega);
        _li_down(gctr_smem, buf, LI, LJ);
        __syncthreads();
        if (LI > 0){
            type2_cart_kernel_lowmem<0,0>(
                buf, LI-1, LJ, LC,
                ish, jsh, ksh,
                ecpbas, ecploc,
                atm, bas, env, omega);
            _li_up(gctr_smem, buf, LI, LJ);
            __syncthreads();
        }

        const int nfi = (LI+1) * (LI+2) / 2;
        const int nfj = (LJ+1) * (LJ+2) / 2;
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double *gx = pgctr;
            double *gy = pgctr +   nao*nao;
            double *gz = pgctr + 2*nao*nao;
            atomicAdd(gx + i*nao + j, gctr_smem[ij]);
            atomicAdd(gy + i*nao + j, gctr_smem[ij+nfi*nfj]);
            atomicAdd(gz + i*nao + j, gctr_smem[ij+2*nfi*nfj]);
        }
        __syncthreads();
    }
}

__global__
void type2_cart_ipipv_lowmem(double *gctr,
                const int LI, const int LJ, const int LC,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env,
                double *work, const int work_size)
{
    double *omega = work + (size_t)blockIdx.x * work_size;
    constexpr int nfi2_max = (AO_LMAX+3)*(AO_LMAX+4)/2;
    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf1[nfi2_max*nfj_max];
    extern __shared__ double smem[];
    double *buf = smem;

    for (int task_id = blockIdx.x; task_id < ntasks; task_id += gridDim.x){
        const int ish = tasks[task_id];
        const int jsh = tasks[task_id + ntasks];
        const int ksh = tasks[task_id + 2*ntasks];

        const int ioff = ao_loc[ish];
        const int joff = ao_loc[jsh];
        const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
        double *pgctr = gctr + ioff*nao + joff + 9*ecp_id*nao*nao;

        type2_cart_kernel_lowmem<2,0>(
            buf1, LI+2, LJ, LC,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env, omega);

        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_down(buf, buf1, LI+1, LJ);
        __syncthreads();
        _li_down_and_write(pgctr, buf, LI, LJ, nao);
        __syncthreads();

        type2_cart_kernel_lowmem<1,0>(
            buf1, LI, LJ, LC,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env, omega);
        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_up(buf, buf1, LI+1, LJ);
        __syncthreads();
        _li_down_and_write(pgctr, buf, LI, LJ, nao);
        __syncthreads();

        if (LI > 0){
            set_shared_memory(buf, 3*nfi1_max*nfj_max);
            _li_down(buf, buf1, LI-1, LJ);
            __syncthreads();
            _li_up_and_write(pgctr, buf, LI, LJ, nao);
            __syncthreads();
            if (LI > 1){
                type2_cart_kernel_lowmem<0,0>(
                    buf1, LI-2, LJ, LC,
                    ish, jsh, ksh,
                    ecpbas, ecploc,
                    atm, bas, env, omega);
                set_shared_memory(buf, 3*nfi1_max*nfj_max);
                _li_up(buf, buf1, LI-1, LJ);
                __syncthreads();
                _li_up_and_write(pgctr, buf, LI, LJ, nao);
            }
        }
        __syncthreads();
    }
}

__global__
void type2_cart_ipvip_lowmem(double *gctr,
                const int LI, const int LJ, const int LC,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env,
                double *work, const int work_size)
{
    double *omega = work + (size_t)blockIdx.x * work_size;
    constexpr int nfi_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    __shared__ double buf1[nfi1_max*nfj1_max];
    extern __shared__ double smem[];
    double *buf = smem;

    for (int task_id = blockIdx.x; task_id < ntasks; task_id += gridDim.x){
        const int ish = tasks[task_id];
        const int jsh = tasks[task_id + ntasks];
        const int ksh = tasks[task_id + 2*ntasks];

        const int ioff = ao_loc[ish];
        const int joff = ao_loc[jsh];
        const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
        double *pgctr = gctr + ioff*nao + joff + 9*ecp_id*nao*nao;

        type2_cart_kernel_lowmem<1,1>(
            buf1, LI+1, LJ+1, LC,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env, omega);

        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_down(buf, buf1, LI, LJ+1);
        __syncthreads();
        _lj_down_and_write(pgctr, buf, LI, LJ, nao);
        __syncthreads();
        if (LI > 0){
            type2_cart_kernel_lowmem<0,1>(
                buf1, LI-1, LJ+1, LC,
                ish, jsh, ksh,
                ecpbas, ecploc,
                atm, bas, env, omega);
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_up(buf, buf1, LI, LJ+1);
            __syncthreads();
            _lj_down_and_write(pgctr, buf, LI, LJ, nao);
            __syncthreads();
        }

        if (LJ > 0){
            type2_cart_kernel_lowmem<1,0>(
                buf1, LI+1, LJ-1, LC,
                ish, jsh, ksh,
                ecpbas, ecploc,
                atm, bas, env, omega);
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_down(buf, buf1, LI, LJ-1);
            __syncthreads();
            _lj_up_and_write(pgctr, buf, LI, LJ, nao);
            __syncthreads();
            if (LI > 0){
                type2_cart_kernel_lowmem<0,0>(
                    buf1, LI-1, LJ-1, LC,
                    ish, jsh, ksh,
                    ecpbas, ecploc,
                    atm, bas, env, omega);
                set_shared_memory(buf, 3*nfi_max*nfj1_max);
                _li_up(buf, buf1, LI, LJ-1);
                __syncthreads();
                _lj_up_and_write(pgctr, buf, LI, LJ, nao);
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

// Whether the static shared memory of kernel plus dynamic_smem_bytes fits in
// one block on the current device.
static bool type2_fits_shared_memory(const void *kernel, const int dynamic_smem_bytes)
{
    int device;
    int max_smem;
    hipFuncAttributes attr;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess ||
        hipFuncGetAttributes(&attr, kernel) != hipSuccess) {
        return true;
    }
    return attr.sharedSizeBytes + dynamic_smem_bytes <= (size_t)max_smem;
}

// Launch one of the *_lowmem kernels with a grid of resident blocks, each
// with its own omega workspace. LI and LJ are the largest angular momenta
// passed to type2_cart_kernel_lowmem by the kernel.
static int type2_launch_lowmem(const void *kernel, const char *name,
                double *gctr, const int li, const int lj, const int lc,
                const int LI, const int LJ, const int min_smem_size,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    if (ntasks == 0) return 0;
    int smem_size = type2_lowmem_smem_size(LI, LJ, lc);
    if (smem_size < min_smem_size) smem_size = min_smem_size;
    const size_t smem_bytes = smem_size * sizeof(double);
    int work_size = type2_lowmem_work_size(LI, LJ, lc);

    int device, n_cu, blocks_per_cu;
    hipError_t err = hipGetDevice(&device);
    if (err == hipSuccess) {
        err = hipDeviceGetAttribute(&n_cu, hipDeviceAttributeMultiprocessorCount, device);
    }
    if (err == hipSuccess) {
        err = hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (err == hipSuccess) {
        err = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, THREADS, smem_bytes);
    }
    if (err != hipSuccess || blocks_per_cu == 0) {
        fprintf(stderr, "HIP Error in %s setup (li,lj,lc = %d,%d,%d): %s\n",
                name, li, lj, lc, hipGetErrorString(err));
        return 1;
    }
    int nblocks = n_cu * blocks_per_cu;
    if (nblocks > ntasks) nblocks = ntasks;

    double *work;
    err = hipMallocAsync((void **)&work, (size_t)nblocks * work_size * sizeof(double), 0);
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Error in %s workspace (li,lj,lc = %d,%d,%d): %s\n",
                name, li, lj, lc, hipGetErrorString(err));
        return 1;
    }
    void *args[] = {&gctr, (void *)&li, (void *)&lj, (void *)&lc,
                    (void *)&ao_loc, (void *)&nao,
                    (void *)&tasks, (void *)&ntasks,
                    (void *)&ecpbas, (void *)&ecploc,
                    (void *)&atm, (void *)&bas, (void *)&env,
                    &work, &work_size};
    err = hipLaunchKernel(kernel, dim3(nblocks), dim3(THREADS), args, smem_bytes, 0);
    hipError_t free_err = hipFreeAsync(work, 0);
    if (err == hipSuccess) err = free_err;
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Error in %s (li,lj,lc = %d,%d,%d): %s\n",
                name, li, lj, lc, hipGetErrorString(err));
        return 1;
    }
    return 0;
}

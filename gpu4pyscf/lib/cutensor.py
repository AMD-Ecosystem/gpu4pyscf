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

import string
import numpy as np
import cupy
from gpu4pyscf.lib import logger

try:
    import cupy_backends.cuda.libs.cutensor  # NOQA
    from cupyx import cutensor
    from cupy_backends.cuda.libs import cutensor as cutensor_backend
    ALGO_DEFAULT = cutensor_backend.ALGO_DEFAULT
    OP_IDENTITY = cutensor_backend.OP_IDENTITY
    JIT_MODE_NONE = cutensor_backend.JIT_MODE_NONE
    WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_MIN
    #WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_RECOMMENDED
    _tensor_descriptors = {}
except (ImportError, AttributeError):
    cutensor = None
    ALGO_DEFAULT = None
    OP_IDENTITY = None
    JIT_MODE_NONE = None
    WORKSPACE_RECOMMENDED = None

def _auto_create_mode(array, mode):
    if not isinstance(mode, cutensor.Mode):
        mode = cutensor.create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode

def _create_tensor_descriptor(a):
    if any(x == 0 for x in a.strides):
        strides = list(a.strides)
        if strides[0] == 0:
            strides[0] = a.nbytes
        for i, x in enumerate(strides[1:]):
            if x == 0:
                strides[i+1] = strides[i]
        a = cupy.ndarray(a.shape, a.dtype, a.data, strides)
    return cutensor.create_tensor_descriptor(a)

_HIP_MAX_BATCH = 65535

def _hip_einsum(subscripts, *operands, **kwargs):
    '''cupy.einsum, except that a matrix-vector product over more than 65535
    batches is computed as a broadcast multiply and sum.

    cupy.einsum drops size-1 axes and then computes each pairwise contraction
    as a batched matmul (B, L, K) @ (B, K, R). When L or R is 1, rocBLAS may
    run it with a gemv kernel that loops over the batches in steps of 65535
    and returns wrong entries for all but the last step
    (https://github.com/ROCm/rocm-libraries/issues/14819). Splitting the batch
    would avoid the loop, but the multiply and sum is also several times faster
    for these shapes.
    '''
    if (not isinstance(subscripts, str) or kwargs or len(operands) < 2 or
            not all(isinstance(x, cupy.ndarray) for x in operands) or
            np.result_type(*operands).kind not in 'fc'):
        return cupy.einsum(subscripts, *operands, **kwargs)
    sub = subscripts.replace(' ', '')
    inputs, arrow, out = sub.partition('->')
    subs = inputs.split(',')
    if len(subs) != len(operands):
        return cupy.einsum(subscripts, *operands)
    if not arrow:
        out = ''.join(sorted(c for c in set(inputs) - {',', '.'}
                             if inputs.count(c) == 1))
        if '.' in inputs:
            out = '...' + out
    if '.' in sub:
        ranks = [x.ndim - len(s) + 3 for x, s in zip(operands, subs)
                 if '...' in s]
        free = [c for c in string.ascii_letters if c not in sub]
        if (not ranks or (inputs + out).count('.') !=
                3 * (len(ranks) + ('...' in out)) or
                min(ranks) < 0 or max(ranks) > len(free)):
            return cupy.einsum(subscripts, *operands)
        ell = ''.join(free[:max(ranks)])
        subs = [s.replace('...', ell[len(ell) - x.ndim + len(s) - 3:])
                for x, s in zip(operands, subs)]
        out = out.replace('...', ell)
    size = {}
    for s, x in zip(subs, operands):
        if len(set(s)) != len(s) or len(s) != x.ndim:
            return cupy.einsum(subscripts, *operands)
        for c, n in zip(s, x.shape):
            if size.get(c, 1) not in (1, n) and n != 1:
                return cupy.einsum(subscripts, *operands)
            size[c] = max(size.get(c, 1), n)
    if len(set(out)) != len(out) or not set(out) <= set(size):
        return cupy.einsum(subscripts, *operands)

    # Follow cupy.einsum's default order: contract the last two operands,
    # then the result with each preceding operand. cupy.einsum drops the
    # size-1 axes of each operand first.
    dims = [{c for c, n in zip(s, x.shape) if n > 1}
            for s, x in zip(subs, operands)]
    keeps = []
    sb, lb = subs[-1], dims[-1]
    regime = False
    for i in range(len(subs) - 2, -1, -1):
        rest = ''.join(subs[:i]) + out
        keep = out if i == 0 else ''.join(
            c for c in dict.fromkeys(subs[i] + sb) if c in rest)
        regime |= _hip_matvec_batch(dims[i], lb, keep, size) > _HIP_MAX_BATCH
        keeps.append(keep)
        sb, lb = keep, (dims[i] | lb) & set(keep)
    if not regime:
        return cupy.einsum(subscripts, *operands)

    dtype = np.result_type(*operands)
    operands = [x.astype(dtype, copy=False) for x in operands]
    b, sb = operands[-1], subs[-1]
    for i, keep in zip(range(len(subs) - 2, -1, -1), keeps):
        b = _hip_einsum2(operands[i], subs[i], b, sb, keep)
        sb = keep
    return b

def _hip_matvec_batch(la, lb, out, size):
    '''The batch count if cupy.einsum computes the contraction of operands
    with the non-trivial axes la and lb as a batched matrix-vector product,
    otherwise 0'''
    out = set(out)
    la, lb = la & (lb | out), lb & (la | out)
    if not (la & lb) - out or (la - lb and lb - la):
        return 0
    return int(np.prod([size[c] for c in la & lb & out]))

def _hip_einsum2(a, sa, b, sb, out):
    la = {c for c, n in zip(sa, a.shape) if n > 1}
    lb = {c for c, n in zip(sb, b.shape) if n > 1}
    size = dict(zip(sa, a.shape))
    for c, n in zip(sb, b.shape):
        size[c] = max(size.get(c, 1), n)
    if _hip_matvec_batch(la, lb, out, size) <= _HIP_MAX_BATCH:
        return cupy.einsum(f'{sa},{sb}->{out}', a, b)
    idx = [c for c in dict.fromkeys(out + sa + sb)
           if c in out or (c in sa and c in sb)]
    def expand(x, s):
        own = tuple(i for i, c in enumerate(s) if c not in idx)
        if own:
            x = x.sum(axis=own)
            s = ''.join(c for c in s if c in idx)
        x = x.transpose([s.index(c) for c in idx if c in s])
        it = iter(x.shape)
        return x.reshape([next(it) if c in s else 1 for c in idx])
    c = expand(a, sa) * expand(b, sb)
    return c.sum(axis=tuple(range(len(out), len(idx))))

def _contract_einsum(pattern, a, b, alpha, beta, out=None, einsum=cupy.einsum):
    if out is None:
        out = cupy.ascontiguousarray(einsum(pattern, a, b))
        out *= alpha
    elif beta == 0.:
        out[:] = einsum(pattern, a, b)
        out *= alpha
    else:
        out *= beta
        tmp = einsum(pattern, a, b)
        tmp *= alpha
        out += tmp
    return out

def contraction(
    pattern, a, b, alpha, beta,
    out=None,
    op_a=OP_IDENTITY,
    op_b=OP_IDENTITY,
    op_c=OP_IDENTITY,
    algo=ALGO_DEFAULT,
    jit_mode=JIT_MODE_NONE,
    compute_desc=0,
    ws_pref=WORKSPACE_RECOMMENDED
):
    if a.size == 0 or b.size == 0:
        # cutensor does not support the 0-sized operands
        return _contract_einsum(pattern, a, b, alpha, beta, out)

    pattern = pattern.replace(" ", "")
    str_a, rest = pattern.split(',')
    str_b, str_c = rest.split('->')
    key = str_a + str_b
    val = list(a.shape) + list(b.shape)
    shape = {k:v for k, v in zip(key, val)}

    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    if len(mode_c) != len(set(mode_c)):
        raise ValueError('Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    if out is None:
        out = cupy.empty([shape[k] for k in str_c], order='C', dtype=dtype)
    c = out

    if a.size == 0 or b.size == 0 or c.size == 0:
        raise ValueError(f"cutensor contraction doesn't support zero-sized array (a.shape = {a.shape}, b.shape = {b.shape}, expected c.shape = {c.shape})")

    desc_a = _create_tensor_descriptor(a)
    desc_b = _create_tensor_descriptor(b)
    desc_c = _create_tensor_descriptor(c)

    mode_a = _auto_create_mode(a, mode_a)
    mode_b = _auto_create_mode(b, mode_b)
    mode_c = _auto_create_mode(c, mode_c)
    operator = cutensor.create_contraction(
        desc_a, mode_a, op_a, desc_b, mode_b, op_b, desc_c, mode_c, op_c,
        compute_desc)
    plan_pref = cutensor.create_plan_preference(algo=algo, jit_mode=jit_mode)
    ws_size = cutensor_backend.estimateWorkspaceSize(
        cutensor._get_handle().ptr, operator.ptr, plan_pref.ptr, ws_pref)
    plan = cutensor.create_plan(operator, plan_pref, ws_limit=ws_size)
    ws = cupy.empty(ws_size, dtype=np.int8)
    out = c

    alpha = np.asarray(alpha, dtype=dtype)
    beta = np.asarray(beta, dtype=dtype)

    handler = cutensor._get_handle()
    cutensor_backend.contract(handler.ptr, plan.ptr,
                             alpha.ctypes.data, a.data.ptr, b.data.ptr,
                             beta.ctypes.data, c.data.ptr, out.data.ptr,
                             ws.data.ptr, ws_size)
    return out

import os
contract_engine = None
if cutensor is None:
    contract_engine = 'cupy'  # default contraction engine
contract_engine = os.environ.get('CONTRACT_ENGINE', contract_engine)

# override the 'contract' function if einsum is customized or cutensor is not found
if contract_engine is not None:
    einsum = None
    if contract_engine == 'opt_einsum':
        import opt_einsum
        einsum = opt_einsum.contract
    elif contract_engine == 'cuquantum':
        from cuquantum import contract as einsum # type: ignore
    elif contract_engine == 'cupy':
        einsum = _hip_einsum if cupy.cuda.runtime.is_hip else cupy.einsum
    else:
        raise RuntimeError('unknown tensor contraction engine.')

    import warnings
    warnings.warn(f'using {contract_engine} as the tensor contraction engine.')
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        try:
            return _contract_einsum(pattern, a, b, alpha, beta, out, einsum)
        except cupy.cuda.memory.OutOfMemoryError:
            print('Out of memory error caused by cupy.einsum. '
                  'It is recommended to install cutensor to resolve this.')
            raise
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)

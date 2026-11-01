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

import unittest
import numpy
import cupy
from gpu4pyscf.lib.cupy_helper import contract, einsum

class KnownValues(unittest.TestCase):
    def test_contract(self):
        a = cupy.random.rand(10,9,11)
        b = cupy.random.rand(11,7,13)
        c_einsum = einsum('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        c_contract = contract('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

        a = cupy.random.rand(10,10,10,10)
        b = cupy.random.rand(20,20)
        c_einsum = einsum('lkji,jl->ik', a, b[10:20,10:20])
        c_contract = contract('lkji,jl->ik', a, b[10:20,10:20])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

        a = cupy.random.rand(10,10,10,10)
        b = cupy.random.rand(20,20)
        c_einsum = einsum('lkji,jk->il', a, b[10:20,10:20])
        c_contract = contract('lkji,jk->il', a, b[10:20,10:20])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

    def test_complex_valued(self):
        a = cupy.random.rand(10,9,11) + cupy.random.rand(10,9,11)*1j
        b = cupy.random.rand(11,7,13) + cupy.random.rand(11,7,13)*1j
        c_einsum = einsum('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        c_contract = contract('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

    def test_cache(self):
        a = cupy.random.rand(20,20,20,20)
        b = cupy.random.rand(20,20)

        c = contract('ijkl,ik->jl', a, b)
        c_einsum = einsum('ijkl,ik->jl', a, b)
        assert cupy.linalg.norm(c - c_einsum) < 1e-10

        c = contract('ijkl,jl->ik', a, b)
        c_einsum = einsum('ijkl,jl->ik', a, b)
        assert cupy.linalg.norm(c - c_einsum) < 1e-10

    # issue 614
    def test_zero_strides(self):
        a = cupy.ones((3, 3))
        assert contract('ij,ji->', a[0,:,None], a[0,None,:]) == 3

    def test_einsum_fallback_contiguous(self):
        from gpu4pyscf.lib.cutensor import _contract_einsum
        from gpu4pyscf.lib.cupy_helper import hermi_triu, transpose_sum
        ncells, nkpts, nao, comp = 8, 4, 6, 1
        expLkz = cupy.random.rand(ncells, nkpts, 2)
        mat = cupy.random.rand(ncells, comp, nao, nao)
        out = _contract_einsum('lkz,lxpq->kxpqz', expLkz, mat, 1., 0.)
        assert out.flags.c_contiguous
        ref = einsum('lkz,lxpq->kxpqz', expLkz, mat)
        assert cupy.linalg.norm(out - ref) < 1e-12
        v = out.view(numpy.complex128)[:,:,:,:,0].reshape(nkpts*comp, nao, nao)
        hermi_triu(v, 1, inplace=True)

        veff = cupy.random.rand(ncells, nao, nao)
        out = _contract_einsum('Lpq,Lkz->kpqz', veff, expLkz, 1., 0.)
        assert out.flags.c_contiguous
        ref = einsum('Lpq,Lkz->kpqz', veff, expLkz)
        assert cupy.linalg.norm(out - ref) < 1e-12
        v = out.view(numpy.complex128)[:,:,:,0]
        transpose_sum(v)

        out2 = _contract_einsum('Lpq,Lkz->kpqz', veff, expLkz, 2., 0.)
        assert out2.flags.c_contiguous
        assert cupy.linalg.norm(out2 - 2*ref) < 1e-12

    def _hip_einsum_matvec(self, subscripts, *operands):
        # Run cupy.einsum and _hip_einsum with the batch threshold lowered to
        # 3. Return the number of batched matrix-vector products each issues
        # and whether _hip_einsum rerouted the call.
        from unittest import mock
        from gpu4pyscf.lib import cutensor
        matmul = cupy.matmul
        def record(a, b, *args, **kwargs):
            if (a.ndim == 3 and b.ndim == 3 and a.shape[0] > 3 and
                    (a.shape[1] == 1 or b.shape[2] == 1)):
                calls[0] += 1
            return matmul(a, b, *args, **kwargs)
        with mock.patch.object(cutensor, '_HIP_MAX_BATCH', 3), \
             mock.patch.object(cupy, 'matmul', record), \
             mock.patch.object(cutensor, '_hip_einsum2',
                               wraps=cutensor._hip_einsum2) as einsum2:
            calls = [0]
            ref = cupy.einsum(subscripts, *operands)
            n_ref = calls[0]
            calls = [0]
            out = cutensor._hip_einsum(subscripts, *operands)
            n_out = calls[0]
        expected = numpy.einsum(subscripts, *[x.get() for x in operands])
        assert out.shape == expected.shape
        assert out.dtype == ref.dtype
        assert abs(out.get() - expected).max() < 1e-10
        return n_ref, n_out, einsum2.called

    def test_hip_einsum_broadcast_matvec(self):
        # 'a' is contracted in the first step because 'd' has size 1 in the
        # first operand, so the step is a matrix-vector product over 'a'
        a = cupy.random.rand(1, 4)
        b = cupy.random.rand(4, 2, 4, 3)
        c = cupy.random.rand(4, 4, 2)
        n_ref, n_out, rerouted = self._hip_einsum_matvec('da,deab,dac->', a, b, c)
        assert n_ref > 0
        assert n_out == 0
        assert rerouted

    def test_hip_einsum_random(self):
        # _hip_einsum reroutes exactly the calls for which cupy.einsum issues
        # a batched matrix-vector product, and leaves none of them behind
        rng = numpy.random.RandomState(7)
        labels = 'abcdef'
        size = {c: n for c, n in zip(labels, (2, 3, 4, 5, 2, 3))}
        n_matvec = 0
        for _ in range(1000):
            subs = [''.join(rng.permutation(list(labels))[:rng.randint(1, 5)])
                    for _ in range(rng.randint(2, 5))]
            used = ''.join(dict.fromkeys(''.join(subs)))
            if rng.rand() < .3:
                subscripts = ','.join(subs)
            else:
                out = rng.permutation(list(used))[:rng.randint(0, len(used))]
                subscripts = ','.join(subs) + '->' + ''.join(out)
            operands = []
            for s in subs:
                shape = [1 if rng.rand() < .25 else size[c] for c in s]
                x = cupy.asarray(rng.rand(*shape))
                if rng.rand() < .2:
                    x = x + 1j * x
                if rng.rand() < .2:
                    x = cupy.asfortranarray(x)
                operands.append(x)
            n_ref, n_out, rerouted = self._hip_einsum_matvec(subscripts, *operands)
            assert n_out == 0, subscripts
            assert rerouted == (n_ref > 0), subscripts
            n_matvec += n_ref > 0
        assert n_matvec > 20

if __name__ == "__main__":
    print("Full tests for cutensor module")
    unittest.main()

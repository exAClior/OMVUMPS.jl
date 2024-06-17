using OMVUMPS, Test
using TensorKit
using MPSKit
using Yao

@testset "Fix points" begin
    D = 3
    d = 2
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    normalizeMPS!(A)
    l, r = fixedpoints(A)

    @tensor contract_lr[] := l[1; 2] * r[2; 1]
    @test contract_lr[1] ≈ one(eltype(contract_lr)) atol = 1e-10

    @tensor lp[-1; -2] := l[1; 2] * A[2 3 -2] * conj(A[1 3 -1])
    @test lp ≈ l

    @tensor rp[-1; -2] := r[1; 3] * A[-1 2 1] * conj(A[-2 2 3])
    @test rp ≈ r
end

@testset "Gauge Fixing" begin
    D = 3
    d = 2
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    normalizeMPS!(A)
    l, r = fixedpoints(A)

    L, Al = leftOrthonormalize(A)
    R, Ar = rightOrthonormalize(A)

    @tensor Al_id[-1; -2] := Al[1 2 -1] * conj(Al[1 2 -2])
    @test Al_id ≈ id(space(Al_id, 1))

    @tensor Ar_id[-1; -2] := Ar[-1 2 1] * conj(Ar[-2 2 1])
    @test Ar_id ≈ id(space(Ar_id, 1))

    Al, Ac, Ar, C = mixedCanonical(A)

    @tensor Ar_id[-1; -2] := Ar[-1; 1 2] * conj(Ar[-2; 1 2])
    @tensor Al_id[-1; -2] := Al[1 2; -2] * conj(Al[1 2; -1])
    @tensor LHS[-1 -2; -3] := Al[-1 -2; 1] * C[1; -3]
    @tensor RHS[-1 -2; -3] := C[-1; 1] * Ar[1; -2 -3]

    @test Ar_id ≈ id(space(Ar, 1))
    @test Al_id ≈ id(space(Al, 1))
    @test LHS ≈ RHS && RHS ≈ Ac
end



@testset "Transfer Matrix" begin
    d = 2
    D = 3
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    A_his = Tensor(Array(A[]), ℝ^D * ℝ^d * ℝ^D)

    function createTransfermatrix(A)
        @tensor E[-1 -2 -3 -4] := A[-1 1 -3] * A'[-2 1 -4]
        return E
    end

    E_mine = transfer_matrix(A)
    E_his = createTransfermatrix(A_his)
    @test E_mine[] ≈ E_his[]
end
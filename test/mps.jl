using OMVUMPS, Test
using TensorKit
using MPSKit
using Yao

@testset "Fix points" begin
    D = 3
    d = 2
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    normalize!(A)
    l, r = fixedpoints(A)

    @tensor contract_lr[] := l[1; 2] * r[2; 1]
    @test contract_lr[1] ≈ one(eltype(contract_lr)) atol = 1e-10


end



@testset "Transfer Matrix" begin
    d = 2
    D = 3
    h_bar = xxx_ham()
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    ψ = InfiniteMPS([A])
    AL = ψ.AL[]
    E = transfer_matrix(AL)
    U, S, Vd = tsvd(E^2000; trunc=truncdim(1))
    cur_tr = Vd * U * S
    @test real(cur_tr) ≈ one(cur_tr) atol = 1e-10
end
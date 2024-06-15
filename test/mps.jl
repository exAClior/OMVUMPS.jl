using OMVUMPS, Test
using TensorKit
using MPSKit
using Yao

using OMVUMPS: transfer_matrix

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
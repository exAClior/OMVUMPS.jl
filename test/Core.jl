using OMVUMPS, Test
using TensorKit
using MPSKit
using Yao

@testset "hamiltonian" begin
    myham = xxx_ham()

    bell_minus = zeros(2,2)
    bell_minus[1,2] = 1/sqrt(2)
    bell_minus[2,1] = -1/sqrt(2)
    bell_minus = Tensor(bell_minus, ℂ^2 * ℂ^2)

    @tensor h_expec[] := conj(bell_minus[i,j]) * myham[i,j,k,l] * bell_minus[k,l]

    @test real(h_expec[][]) ≈ -0.75
end


@testset "energy" begin
    myham = xxx_ham()
    ghz_A = zeros(2,2,2)
    ghz_A[1,1,1]  = 1
    ghz_A[2,2,2]  = 1

    ψ = InfiniteMPS([TensorMap(ghz_A, ℂ^2 * ℂ^2, ℂ^2)])

    @test energy_density(myham,ψ) ≈ Yao.expect((kron(X,X)+kron(Y,Y)+kron(Z,Z)/4.0),ghz_state(2))


end

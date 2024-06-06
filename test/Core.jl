using OMVUMPS, Test
using TensorKit
using MPSKit
using Yao

@testset "sum Left/Right" begin
    d = 2
    D = 3
    h_bar = xxx_ham()
    A = TensorMap(reshape(rand_unitary(d*D)[1:D,:],D,d,D), ℂ^D * ℂ^d, ℂ^D)
    ψ = InfiniteMPS([A])
    A_gauged = ψ.AL[]

    # for AL in [ψ.AL[], ψ.AR[]]
        AL = ψ.AL[]
        @tensor ALH[i,j] := AL[][f,a,e] * AL[][i,b,f] * h_bar[][a,b,c,d] * conj(AL[][g,c,e]) * conj(AL[][j,d,g])
        ALH = reshape(ALH,(reduce(*,size(ALH))))
        @tensor ELL[i,j;k,l] := AL[][i,a,k] * conj(AL[][j,a,l])
        ELL = reshape(ELL, D^2,D^2)
        Lh = sumLeft(AL,h_bar,1e-8)
        prev_term = copy(ALH)
        prev_term = ELL * prev_term
        sum_Cur = copy(prev_term)
        for _ in 1:100
            prev_term = ELL * prev_term
            sum_Cur = sum_Cur + prev_term
        end
        @show sum_Cur
        Lh
        @show Lh - sum_Cur

    # end
end

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

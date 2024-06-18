using OMVUMPS, Test
using TensorKit
using Yao

@testset "VUMPS" begin
    myham = xxx_ham()
    d = 2
    D = 137
    A0 = TensorMap(randn, ComplexF64, ℂ^D * ℂ^d, ℂ^D)
    A0 = normalize(A0)
    vumps(myham, A0, tol=1e-4, tolFactor=1e-2, verbose=true)
end



@testset "energy" begin
    myham = xxx_ham()
    ghz_A = zeros(2, 2, 2)
    ghz_A[1, 1, 1] = one(ComplexF64)
    ghz_A[2, 2, 2] = one(ComplexF64)

    Tensor
    Al, Ac, Ar, C = mixedCanonical(TensorMap(ghz_A, ℂ^2 * ℂ^2, ℂ^2))
    @test energy_density(myham, Al, Ac) ≈ Yao.expect((kron(X, X) + kron(Y, Y) + kron(Z, Z) / 4.0), ghz_state(2))

    myham_reg = OMVUMPS.regularize(myham, Al, Ac)
    @test energy_density(myham_reg, Al, Ac) ≈ zero(real(eltype(myham_reg)))
end

@testset "create hamiltonian" begin
    myham = xxx_ham()

    bell_minus = zeros(2, 2)
    bell_minus[1, 2] = 1 / sqrt(2)
    bell_minus[2, 1] = -1 / sqrt(2)
    bell_minus = Tensor(bell_minus, ℂ^2 * ℂ^2)

    @tensor h_expec[] := conj(bell_minus[i, j]) * myham[i, j, k, l] * bell_minus[k, l]

    @test real(h_expec[][]) ≈ -0.75
end



@testset "H AC/L" begin
    d = 2
    D = 3
    h_bar = xxx_ham()
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    ψ = InfiniteMPS([A])
    AL = ψ.AL[]
    AR = ψ.AR[]

    hal = h_expect_L(xxx_ham(), AL)
    har = h_expect_R(xxx_ham(), AR)
    tr(hal)
    tr(har)
    # @test tr(hal) ≈ tr(har)
end

@testset "sum Left/Right" begin
    d = 2
    D = 3
    h_bar = xxx_ham()
    A = TensorMap(reshape(rand_unitary(d * D)[:, 1:D], D, d, D), ℂ^D * ℂ^d, ℂ^D)
    ψ = InfiniteMPS([A])
    A_gauged = ψ.AL[]

    # for AL in [ψ.AL[], ψ.AR[]]
    AL = ψ.AL[]
    ALH = h_expect_L(h_bar, AL)
    ALH = permute(ALH, (), (1, 2))
    EL = transfer_matrix(AL)

    sum_EL = one(EL)
    for i in 1:100
        sum_EL += EL^i
    end

    HL = sum_EL * ALH'
    Lh = sumLeft(AL, h_bar, 1e-8)
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




function xxx_ham()
    Sx = TensorMap(ComplexF64[0 1; 1 0] / 2.0, ℂ^2 ← ℂ^2)
    Sy = TensorMap(ComplexF64[0 -im; im 0] / 2.0, ℂ^2 ← ℂ^2)
    Sz = TensorMap(ComplexF64[1 0; 0 -1] / 2.0, ℂ^2 ← ℂ^2)
    return (Sx ⊗ Sx + Sy ⊗ Sy + Sz ⊗ Sz)
end

function tfi_ham()
    Sx = TensorMap(ComplexF64[0 1; 1 0] / 2.0, ℂ^2 ← ℂ^2)
    Sz = TensorMap(ComplexF64[1 0; 0 -1] / 2.0, ℂ^2 ← ℂ^2)
    I = id(ℂ^2)
    return -(Sx ⊗ Sx) - (Sz ⊗ I + I ⊗ Sz) / 2.0
end

function energy_density(h::AbstractTensorMap, AL::AbstractTensorMap, AC::AbstractTensorMap)
    # implements (35)
    # AL( b * i,a)
    # b -- AL -- a -- AC --d
    # b    i          j    d
    # b          h         d
    # b    k          l    d
    # b -- AL'-- c -- AC'--d 
    @tensor energy[] := conj(AL[b, k, c]) * conj(AC[c, l, d]) * h[k, l, i, j] * AL[b, i, a] * AC[a, j, d]

    return real(energy[][])
end

function regularize(h::TensorMap, AL::TensorMap, AC::TensorMap)
    exp_val = energy_density(h, AL, AC)
    return h - exp_val * id(domain(h))
end

function Ẽright(v::TensorMap, A::TensorMap, fpts=fixedpoints(A))
    l, r = fpts

    @tensor transfer[-1; -2] := A[-1 2 1] * conj(A[-2 2 3]) * v[1; 3]
    fixed = tr(l * v) * r
    vNew = v - transfer + fixed
    return vNew
end

function Ẽleft(v::TensorMap, A::TensorMap, fpts=fixedpoints(A))
    l, r = fpts

    @tensor transfer[-1; -2] := A[1 2 -2] * conj(A[3 2 -1]) * v[3; 1]
    fixed = tr(v * r) * l
    vNew = v - transfer + fixed
    return vNew
end

function LhMixed(h̃::TensorMap, Al::TensorMap, C::TensorMap; tol::AbstractFloat=1e-5)
    tol = max(tol, 1e-14)

    l = id(space(Al, 1))
    r = C * C'

    @tensor b[-1; -2] := Al[4 2; 1] * Al[1 3; -2] * conj(Al[4 5; 6]) * conj(Al[6 7; -1]) * h̃[5 7; 2 3]

    Lh, _ = linsolve(v -> Ẽleft(v, Al, (l, r)), b; tol)

    return Lh
end


function RhMixed(h̃, Ar, C; tol::AbstractFloat=1e-5)
    tol = max(tol, 1e-14)

    r = id(space(Ar, 1))
    l = C' * C

    @tensor b[-1; -2] := Ar[-1; 2 1] * Ar[1; 3 4] * conj(Ar[-2; 7 6]) * conj(Ar[6; 5 4]) * h̃[5 7; 2 3]

    Rh, _ = linsolve(v -> Ẽright(v, Ar, (l, r)), b; tol)

    return Rh
end

function H_Ac(v, h̃, Al, Ar, Lh, Rh)
    # the function that applies (131) to input tensor v in the place of Ac
    @tensor term1[-1 -2; -3] := Al[4 2; 1] * v[1 3; -3] * conj(Al[4 5; -1]) * h̃[5 -2; 2 3]

    @tensor term2[-1 -2; -3] := v[-1 2; 1] * Ar[1; 3 4] * conj(Ar[-3; 5 4]) * h̃[-2 5; 2 3]

    @tensor term3[-1 -2; -3] := Lh[-1; 1] * v[1 -2; -3]

    @tensor term4[-1 -2; -3] := v[-1 -2; 1] * Rh[1; -3]

    return term1 + term2 + term3 + term4
end

function H_C(v, h̃, Al, Ar, Lh, Rh)
    @tensor term1[-1; -2] := Al[5 3; 1] * v[1; 2] * Ar[2; 4 7] * conj(Al[5 6; -1]) * conj(Ar[-2; 8 7]) * h̃[6 8; 3 4]

    term2 = Lh * v

    term3 = v * Rh

    return term1 + term2 + term3
end

function calcNewCenter(h̃, Al, Ac, Ar, C; tol::AbstractFloat=1e-5, Lh=LhMixed(h̄, Al, C; tol=max(tol, 1e-15)), Rh=RhMixed(h̄, Ar, C; tol=max(tol, 1e-15)))
    tol = max(tol, 1e-14)

    _, vecs, _ = eigsolve(v -> H_Ac(v, h̃, Al, Ar, Lh, Rh), Ac, 1, :SR; tol)
    Ãc = vecs[1]


    _, vecs, _ = eigsolve(v -> H_C(v, h̃, Al, Ar, Lh, Rh), C, 1, :SR; tol)
    C̃ = vecs[1]

    return Ãc, C̃
end

function minAcC(Ãc, C̃; tol::AbstractFloat=1e-5)
    tol = max(tol, 1e-14)

    UlAc, _ = leftorth(Ãc, (1, 2), (3,); alg=Polar())

    UlC, _ = leftorth(C̃, (1,), (2,); alg=Polar())

    Al = UlAc * UlC'

    C, Ar = rightOrthonormalize(Al, C̃; tol)
    nrm = tr(C * C')
    C /= sqrt(nrm)
    @tensor Ac[-1 -2; -3] := Al[-1 -2; 1] * C[1; -3]

    return Al, Ac, Ar, C
end

function gradientNorm(h̃, Al, Ac, Ar, C, Lh, Rh)
    AcUpdate = H_Ac(Ac, h̃, Al, Ar, Lh, Rh)
    CUpdate = H_C(C, h̃, Al, Ar, Lh, Rh)
    @tensor AlCupdate[-1 -2; -3] := Al[-1 -2; 1] * CUpdate[1; -3]

    return norm(AcUpdate - AlCupdate)
end

function vumps(h, A0; tol::AbstractFloat=1e-4, tolFactor::AbstractFloat=1e-1, verbose::Bool=true)

    Al, Ac, Ar, C = mixedCanonical(A0)

    flag = true
    delta = 1e-5
    i = 0

    while flag && i < 10000
        i += 1

        h̃ = regularize(h, Al, Ac)

        Lh = LhMixed(h̃, Al, C; tol=delta * tolFactor)
        Rh = RhMixed(h̃, Ar, C; tol=delta * tolFactor)

        delta = gradientNorm(h̃, Al, Ac, Ar, C, Lh, Rh)

        delta < tol && (flag = false)

        Ãc, C̃ = calcNewCenter(h̃, Al, Ac, Ar, C; tol=delta * tolFactor, Lh=Lh, Rh=Rh)

        Ãl, Ãc, Ãr, C̃ = minAcC(Ãc, C̃; tol=delta * tolFactor^2)

        Al, Ac, Ar, C = Ãl, Ãc, Ãr, C̃

        # print current energy
        if verbose
            E = real(energy_density(h, Al, Ac))
            println("iteration:\t$(i)\tenergy:\t$(E)\tgradient norm:\t$(delta)\n")
        end
    end
    E = real(energy_density(h, Al, Ac))

    return E, Al, Ac, Ar, C
end
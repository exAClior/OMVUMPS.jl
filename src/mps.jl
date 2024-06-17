using TensorKit, KrylovKit

function transfer_matrix(A::TensorMap)
    # k---A---i        k-----------i
    #     a        =         E
    # l---A*---j       l-----------j 
    @tensor E[-1 -2; -3 -4] := A[-1, 1, -3] * conj(A[-2, 1, -4])
    return E
end

function normalizeMPS!(A::TensorMap)
    evals, _, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), 1, :LM) do v
            @tensor v_out[-1; -2] := A[-1 2 1] * conj(A[-2 2 3]) * v[1; 3]
        end
    A /= sqrt(evals[1])
    return A
end

function normalizeMPS(A::TensorMap)
    A_copy = copy(A)
    return normalizeMPS!(A_copy)
end

function leftFixedPoint(A::TensorMap)
    E = transfer_matrix(A)

    _, eigvecs, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), 1, :LM) do v
            @tensor v_out[-1; -2] := v[2; 1] * E[1 2; -2 -1]
        end

    l = eigvecs[1]
    tracel = tr(l)
    l /= (tracel / abs(tracel))
    l = (l + l') / 2
    return l
end

function rightFixedPoint(A::TensorMap)
    E = transfer_matrix(A)

    _, eigvecs, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), 1, :LM) do v
            @tensor v_out[-1; -2] := E[-1 -2; 1 2] * v[1; 2]
        end

    r = eigvecs[1]
    tracer = tr(r)
    r /= (tracer / abs(tracer))
    r = (r + r') / 2
    return r
end

function fixedpoints(A::TensorMap)
    l, r = leftFixedPoint(A), rightFixedPoint(A)

    trace = tr(l * r)
    return l / trace, r
end

function leftOrthonormalize(A::TensorMap, Lprev::TensorMap=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)); tol::Float64=1e-14, maxiter::Int64=100000)
    tol = max(tol, 1e-14)

    Lprev /= norm(Lprev)
    convergence = false
    Al = similar(A)
    L = similar(Lprev)

    for i in 1:maxiter
        @tensor Ai[-1 -2 -3] := Lprev[-1; 1] * A[1 -2 -3]
        Al, L = leftorth(Ai, (1, 2), (3,); alg=QRpos())
        L /= norm(L)

        convergence = norm(L - Lprev) < tol
        convergence && break
        if i == maxiter
            error("leftOrthonormalize did not converge")
        end
        Lprev = L
    end
    return L, Al
end


function rightOrthonormalize(A::TensorMap, Rprev::TensorMap=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)); tol::Float64=1e-14, maxiter::Int64=100000)
    tol = max(tol, 1e-14)

    Rprev /= norm(Rprev)
    convergence = false
    Ar = similar(A)
    R = similar(Rprev)

    for i in 1:maxiter
        @tensor Ai[-1 -2 -3] := A[-1 -2 1] * Rprev[1; -3]
        R, Ar = rightorth(Ai, (1,), (2, 3); alg=LQpos())
        R /= norm(R)

        convergence = norm(R - Rprev) < tol
        convergence && break
        if i == maxiter
            error("rightOrthonormalize did not converge")
        end

        Rprev = R
    end
    return R, Ar
end

function mixedCanonical(A::TensorMap; L0::TensorMap=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), R0::TensorMap=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), tol::Float64=1e-14, maxiter::Int64=100000)
    tol = max(1e-14, tol)

    L, Al = leftOrthonormalize(A, L0; tol=tol, maxiter=maxiter)
    R, Ar = rightOrthonormalize(A, R0; tol=tol, maxiter=maxiter)

    C = L * R

    U, C, Vdg = tsvd(C, (1,), (2,))

    @tensor Al[-1 -2; -3] = conj(U[1; -1]) * Al[1 -2; 2] * U[2; -3]
    @tensor Ar[-1; -2 -3] = Vdg[-1; 1] * Ar[1; -2 2] * conj(Vdg[-3; 2])

    Cnorm = tr(C * C')
    C /= sqrt(Cnorm)

    @tensor Ac[-1 -2; -3] := Al[-1 -2; 1] * C[1; -3]

    return Al, Ac, Ar, C
end
function transfer_matrix(A::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    # -1---A----3        -1------------3
    #      1        =           E
    # -2---A*----4       -2------------4 
    @tensor E[-1 -2; -3 -4] := A[-1 1 -3] * conj(A[-2 1 -4])
    return E
end

function normalizeMPS!(A::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    evals, _, _ =
        eigsolve(TensorMap(randn, eltype(A), codomain(A, 1) ← domain(A, 1)), 1, :LM) do v
            @tensor v_out[-1; -2] := A[-1 2 1] * conj(A[-2 2 3]) * v[1; 3]
        end
    A /= sqrt(evals[1])
    return A
end

function normalizeMPS(A::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    A_copy = copy(A)
    return normalizeMPS!(A_copy)
end

myrandisometry(dims::Base.Dims{2}) = randisometry(Float64, dims)
function myrandisometry(::Type{T}, dims::Base.Dims{2}) where {T<:Number}
    return dims[1] >= dims[2] ? rand_unitary(dims[1])[:, 1:dims[2]] :
           throw(DimensionMismatch("cannot create isometric matrix with dimensions $dims; isometry needs to be tall or square"))
end

function leftFixedPoint(A::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    E = transfer_matrix(A)

    _, eigvecs, _ =
        eigsolve(TensorMap(randn, eltype(A), codomain(A, 1) ← domain(A, 1)), 1, :LM) do v
            @tensor v_out[-1; -2] := v[2; 1] * E[1 2; -2 -1]
        end

    l = eigvecs[1]
    tracel = tr(l)
    l /= (tracel / abs(tracel))
    l = (l + l') / 2
    return l
end

function rightFixedPoint(A::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    E = transfer_matrix(A)

    _, eigvecs, _ =
        eigsolve(TensorMap(randn, eltype(A), codomain(A, 1) ← domain(A, 1)), 1, :LM) do v
            @tensor v_out[-1; -2] := E[-1 -2; 1 2] * v[1; 2]
        end

    r = eigvecs[1]
    tracer = tr(r)
    r /= (tracer / abs(tracer))
    r = (r + r') / 2
    return r
end

function fixedpoints(A::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    l, r = leftFixedPoint(A), rightFixedPoint(A)
    trace = tr(l * r)
    return l / trace, r
end

function leftOrthonormalize(A::AbstractTensorMap, Lprev::AbstractTensorMap=TensorMap(randn, eltype(A), codomain(A, 1) ← domain(A, 1)); tol::Float64=1e-14, maxiter::Int64=100000)
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


function rightOrthonormalize(A::AbstractTensorMap, Rprev::AbstractTensorMap=TensorMap(randn, eltype(A), codomain(A, 1) ← domain(A, 1)); tol::Float64=1e-14, maxiter::Int64=100000)
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

function mixedCanonical(A::AbstractTensorMap; L0::AbstractTensorMap=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), R0::TensorMap=TensorMap(randn, eltype(A), codomain(A, 1) ← domain(A, 1)), tol::Float64=1e-14, maxiter::Int64=100000)
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
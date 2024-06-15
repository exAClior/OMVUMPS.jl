using TensorKit, MPSKitModels
using MPSKit
using Yao, LinearAlgebra

# d = 2
# D =3

# A = TensorMap(randn,ℂ^2*ℂ^3,ℂ^4)

# @tensor testT[i;j] := A[][a,b,i] * conj(A[][a,b,j])
# @tensor testT[i,j] := A[][a,b,i] * conj(A[][a,b,j])

# @tensor testT[i,j,k] := A[i,j,k] 
# @tensor testT[i,j,k] := A[j,i,k] 
# @tensor testT[i,j,k] := A[i,k,j] 

# @tensor testT[i,j,k] := A'[i,j,k] 
# @tensor testT[i,j,k] := A'[j,i,k] 
# @tensor testT[i,j,k] := A'[i,k,j] 
# @tensor testT[i,j] := A[a b i] * A'[j a b]
# @tensor testT[i,j] := A[a i b] * A'[b a j]
# @tensor testT[i,j] := A[i b a] * A'[a j b]

# A_mat = rand_unitary(d*D)[:,1:D]
# A_mat' * A_mat
# A = TensorMap(A_mat,ℂ^D*ℂ^d,ℂ^D)
# A'

# ψ = InfiniteMPS([A])

# AL = ψ.AL[1]
# AR = ψ.AR[1]
# AL' * AL
# AR' * AR


# @tensor testT[j,i] := A[a,b,i] * A'[j,a,b]


# @tensor testT[j,i] := AL[a,b,i] * AL'[j,a,b]
# @tensor testT[j,i] := AR[j,a,b] * AR'[b,i,a]

function xxx_ham()
    local_term = Array(reshape(mat((kron(X, X) + kron(Y, Y) + kron(Z, Z)) / 4.0), 2, 2, 2, 2))
    return TensorMap(local_term, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)
end


function energy_density(h::TensorMap, ψ::InfiniteMPS)
    # implements (35)
    # AL( b * i,a)
    # b -- AL -- a -- AC --d
    # b    i          j    d
    # b          h         d
    # b    k          l    d
    # b -- AL'-- c -- AC'--d 
    AL = ψ.AL[]
    AC = ψ.AC[]
    @tensor energy[] := conj(AL[b, k, c]) * conj(AC[c, l, d]) * h[k, l, i, j] * AL[b, i, a] * AC[a, j, d]

    # AR = ψ.AL[]
    # @tensor energy2[] := conj(AR[c,l,d]) * conj(AC[b,k,c]) * h[k,l,i,j] * AR[a,j,d] * AC[b,i,a]
    # @assert energy ≈ energy2
    # @show energy, energy2

    return real(energy[][])
end


function h_expect_R(h_bar::TensorMap, AR::TensorMap)
    # j--AR---f--AR----|
    #     a       b    |
    #         h        e
    #     c       d    |
    # i--AR*--g--AR*---|

    # j== AR  =b=   AR' == i  is  j ==== i
    #     |--- a ---|

    @tensor ARH[j; i] := AR[j, a, f] * AR[f, b, e] * h_bar[c, d, a, b] * conj(AR[i, c, g]) * conj(AR[g, d, e])
    return ARH
end

function h_expect_L(h_bar::TensorMap, AL::TensorMap)
    # ---AL--f--AL---i
    # |   a      b
    # e      h
    # |   c      d
    # ---AL*--g--AL*---j

    # j== AL' = a =  AL == i  is  j ==== i
    #      |--- b ----|

    @tensor ALH[j; i] := AL[e, a, f] * AL[f, b, i] * h_bar[c, d, a, b] * conj(AL[e, c, g]) * conj(AL[g, d, j])
    return ALH
end

function sumLeft(AL::TensorMap, h_bar::TensorMap, tol::Float64)
    EL = transfer_matrix(AL)
    ALH = h_expect_L(h_bar, AL)
    Lh = ALH / (one(EL) - EL)
    return Lh
end

function sumRight(AR::TensorMap, h_bar::TensorMap, tol::Float64)
    ER = transfer_matrix(AR)
    ARH = h_expect_R(h_bar, AR)
    Rh = (one(ER) - ER) \ ARH
    return Rh
end

# function VUMPS(h::TensorMap, A::TensorMap,η::T) where {T}
#     ψ = InfiniteMPS([A])
#     shift_h = energy_density(h,ψ)
#     h_bar = h - shift_h * TensorMap(diagm(ones(T,reduce(*,dims(domain(h))))), codomain(h), domain(h))
#     return nothing
# end

# H = heisenberg_XXX(;spin = 1//2)

# D = 137 
# Ψ = InfiniteMPS(ℂ^2, ℂ^D)
# algorithm1 = VUMPS(;tol_galerkin=1e-10,maxiter=100)
# Ψ₀, envs = find_groundstate(Ψ, H, algorithm1);

# algorithm2 = GradientGrassmann(;tol=1e-15,maxiter=300)
# Ψ₀1, envs = find_groundstate(Ψ, H, algorithm2);
# expectation_value(Ψ₀1,H)

# H = heisenberg_XYZ()
# Ψ = InfiniteMPS(ℂ^3, ℂ^D)
# algorithm = VUMPS(;tol_galerkin=1e-10,dynamical_tols=false, maxiter=1000)
# Ψ₀, envs = find_groundstate(Ψ, H, algorithm);

# expectation_value(Ψ₀,H)

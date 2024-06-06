using TensorKit, MPSKitModels
using MPSKit
using Yao, LinearAlgebra

function xxx_ham()
    local_term = Array(reshape(mat((kron(X,X) + kron(Y,Y) + kron(Z,Z)) /4.0), 2, 2, 2, 2))
    return TensorMap(local_term, ℂ^2  * ℂ^2, ℂ^2 * ℂ^2)
end


function energy_density(h::TensorMap, ψ::InfiniteMPS)
    # implements (35)
    AL = ψ.AL[]
    AC = ψ.AC[]
    @tensor energy[] := conj(AL[d,i,a]) * conj(AC[c,j,d]) * h[i,j,k,l] * AL[b,k,a] * AC[c,l,b]
    return real(energy[][])
end

function sumLeft(AL::TensorMap,h_bar::TensorMap, tol::Float64)
    # i---AL---k        i-----------k
    #      a      =           EL
    # j---AL*---l       j-----------l 

    @tensor EL[i,j;k,l] := AL[][i,a,k] * conj(AL[][j,a,l])
    EL = reshape(EL, (reduce(*,size(EL)[1:2]),reduce(*,size(EL)[3:4])))

    # ---AL--f--AL---i
    # |   a      b
    # e      h
    # |   c      d
    # ---AL*--g--AL*---j

    @tensor ALH[i,j] := AL[][f,a,e] * AL[][i,b,f] * h_bar[][a,b,c,d] * conj(AL[][g,c,e]) * conj(AL[][j,d,g])
    ALH = reshape(ALH,(reduce(*,size(ALH))))

    Lh = (IMatrix(EL)-EL)\ALH
    return Lh
end

# function VUMPS(h::TensorMap, A::TensorMap,η::T) where {T}
#     ψ = InfiniteMPS([A])
#     shift_h = energy_density(h,ψ)
#     h_bar = h - shift_h * TensorMap(diagm(ones(T,reduce(*,dims(domain(h))))), codomain(h), domain(h))
#     return nothing
# end

# H = heisenberg_XXX(;spin = 1//2)

# using MPSKit, MPSKitModels
# D = 30
# Ψ = InfiniteMPS(ℂ^2, ℂ^D)
# algorithm = VUMPS()
# Ψ₀, envs = find_groundstate(Ψ, H, algorithm);

# H = heisenberg_XYZ()

# Ψ = InfiniteMPS(ℂ^3, ℂ^D)
# algorithm = VUMPS()
# Ψ₀, envs = find_groundstate(Ψ, H, algorithm);

# expectation_value(Ψ₀,H)

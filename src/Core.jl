using TensorKit, MPSKitModels
using MPSKit
using Yao, LinearAlgebra

function xxx_ham()
    local_term = Array(reshape(mat((kron(X,X) + kron(Y,Y) + kron(Z,Z)) /4.0), 2, 2, 2, 2))
    return TensorMap(local_term, ℂ^2  * ℂ^2, ℂ^2 * ℂ^2)
end

function energy_density(h::TensorMap, ψ::InfiniteMPS)
    AL = ψ.AL[]
    AC = ψ.AC[]
    @tensor energy[] := conj(AL[d,i,a]) * conj(AC[c,j,d]) * h[i,j,k,l] * AL[b,k,a] * AC[c,l,b]
    return real(energy[][])
end

function sumLeft(AL::TensorMap,h_bar::TensorMap, tol::Float64)

end

function sumRight(AR::TensorMap,h_bar::TensorMap, tol::Float64)

end

function VUMPS(h::TensorMap, A::TensorMap,η::T) where {T}
    ψ = InfiniteMPS([A])
    shift_h = energy_density(h,ψ)
    h_bar = h - shift_h * TensorMap(diagm(ones(T,reduce(*,dims(domain(h))))), codomain(h), domain(h))
    return nothing
end

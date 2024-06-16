using TensorKit, KrylovKit

function transfer_matrix(A::TensorMap)
    # k---A---i        k-----------i
    #     a        =         E
    # l---A*---j       l-----------j 
    @tensor E[-1 -2; -3 -4] := A[-1, 1, -3] * conj(A[-2, 1, -4])
    return E
end

function normalize!(A::TensorMap)
    evals, _, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← dual(space(A, 3))), 1, :LM) do v
            @tensor v_out[-1; -2] := A[-1 2 1] * conj(A[-2 2 3]) * v[1; 3]
        end
    A /= sqrt(evals[1])
    return A
end

function normalize(A::TensorMap)
    A_copy = copy(A)
    return normalize!(A_copy)
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



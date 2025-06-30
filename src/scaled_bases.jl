export ConstantScaledBasis

"""
Scale each basis elements by vector of constants, e.g., ̃p_j(x) = v[j] * p_j(x)
"""
struct ConstantScaledBasis{B<:UnivariateBasis, V<:AbstractVector{<:Real}} <: UnivariateBasis
    basis::B
    scaling::V
end

function Evaluate!(space::AbstractMatrix{U}, basis::ConstantScaledBasis, x::AbstractVector{U}) where {U}
    @argcheck size(space, 1) <= length(basis.scaling) DimensionMismatch
    Evaluate!(space, basis.basis, x)
    scaling = basis.scaling
    AK.foreachindex(space) do lin_idx; @inbounds begin
        deg, idx = (lin_idx - 1) ÷ size(space, 2) + 1, (lin_idx - 1) % size(space, 2) + 1
        space[deg, idx] *= scaling[deg]
    end; end
    nothing
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, basis::ConstantScaledBasis, x::AbstractVector{U}) where {U}
    p, N = size(eval_space)
    @argcheck p <= length(basis.scaling) DimensionMismatch
    EvalDiff!(eval_space, diff_space, basis.basis, x)
    scaling = basis.scaling
    AK.foreachindex(eval_space) do lin_idx; @inbounds begin
        deg, idx = (lin_idx - 1) ÷ N + 1, (lin_idx - 1) % N + 1
        eval_space[deg, idx] *= scaling[deg]
        diff_space[deg, idx] *= scaling[deg]
    end; end
    nothing
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, diff2_space::AbstractMatrix{U}, basis::ConstantScaledBasis, x::AbstractVector{U}) where {U}
    @argcheck size(eval_space, 1) <= length(basis.scaling) DimensionMismatch
    EvalDiff2!(eval_space, diff_space, diff2_space, basis.basis, x)
    scaling = basis.scaling
    AK.foreachindex(eval_space) do lin_idx; @inbounds begin
        deg, idx = (lin_idx - 1) ÷ size(eval_space, 2) + 1, (lin_idx - 1) % size(eval_space, 2) + 1
        eval_space[deg, idx] *= scaling[deg]
        diff_space[deg, idx] *= scaling[deg]
        diff2_space[deg, idx] *= scaling[deg]
    end; end
    nothing
end
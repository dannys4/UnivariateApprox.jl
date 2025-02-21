export Evaluate, Evaluate!
export EvaluateDegree, EvaluateDegree!
export EvalDiff, EvalDiff!
export EvalDiff2, EvalDiff2!
export ScaledBasis

abstract type UnivariateBasis end

"""
    Evaluate!(space::AbstractMatrix, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` at `x` and store the result in `space`.

# Example
```jldoctest
julia> space = zeros(3,2);

julia> Evaluate!(space, LegendrePolynomial(), [0.5, 0.75])

julia> space
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375
```

See also: [`Evaluate`](@ref)
"""
Evaluate!


"""
    EvaluateDegree!(space::AbstractVector, degree::Int, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` of exactly `degree` at `x` and store the result in `space`.
"""
function EvaluateDegree!(space::AbstractVector{U}, degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    full_evals = Evaluate(degree, basis, x)
    @. space = full_evals[end, :]
    nothing
end


"""
    EvalDiff!(eval_space::AbstractMatrix, diff_space::AbstractMatrix, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` and its derivative at `x` and store the results in `eval_space` and `diff_space`, respectively.

# Example
```jldoctest
julia> eval_space = zeros(3,2);

julia> diff_space = zeros(3,2);

julia> EvalDiff!(eval_space, diff_space, LegendrePolynomial(), [0.5, 0.75])

julia> eval_space
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375

julia> diff_space
3×2 Matrix{Float64}:
 0.0  0.0
 1.0  1.0
 1.5  2.25
```

See also: [`EvalDiff`](@ref)
"""
EvalDiff!

"""
    EvalDiff2!(eval_space::AbstractMatrix, diff_space::AbstractMatrix, diff2_space::AbstractMatrix, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` and its first two derivatives at `x` and store the results in `eval_space`, `diff_space` and `diff2_space`, respectively.

# Example
```jldoctest
julia> eval_space = zeros(3,2);

julia> diff_space = zeros(3,2);

julia> diff2_space = zeros(3,2);

julia> EvalDiff2!(eval_space, diff_space, diff2_space, LegendrePolynomial(), [0.5, 0.75])

julia> eval_space
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375

julia> diff_space
3×2 Matrix{Float64}:
 0.0  0.0
 1.0  1.0
 1.5  2.25

julia> diff2_space
3×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
 3.0  3.0
```

See also: [`EvalDiff2`](@ref)
"""
EvalDiff2!

"""
    Evaluate(max_degree::Int, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` at `x` and return the result.

# Example
```jldoctest
julia> Evaluate(2, LegendrePolynomial(), [0.5, 0.75])
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375
```

See also: [`Evaluate!`](@ref)
"""
function Evaluate(max_degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    space = similar(x, (max_degree + 1, length(x)))
    Evaluate!(space, basis, x)
    space
end

"""
    EvaluateDegree(degree::Int, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` of exactly `degree` at `x` and return the result.
"""
function EvaluateDegree(degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    space = similar(x)
    EvaluateDegree!(space, degree, basis, x)
    space
end

"""
    EvalDiff(max_degree::Int, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` and its derivative at `x` and return the result.

# Example
```jldoctest
julia> eval_space, diff_space = EvalDiff(2, LegendrePolynomial(), [0.5, 0.75]);

julia> eval_space
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375

julia> diff_space
3×2 Matrix{Float64}:
 0.0  0.0
 1.0  1.0
 1.5  2.25
```

See also: [`EvalDiff!`](@ref)
"""
function EvalDiff(max_degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    eval_space = similar(x, (max_degree + 1, length(x)))
    diff_space = similar(x, (max_degree + 1, length(x)))
    EvalDiff!(eval_space, diff_space, basis, x)
    eval_space, diff_space
end

"""
    EvalDiff2(max_degree::Int, basis::UnivariateBasis, x::AbstractVector)

Evaluate the univariate basis `basis` and its first two derivatives at `x` and return the results.

# Example
```jldoctest
julia> eval_space, diff_space, diff2_space = EvalDiff2(2, LegendrePolynomial(), [0.5, 0.75]);

julia> eval_space
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375

julia> diff_space
3×2 Matrix{Float64}:
 0.0  0.0
 1.0  1.0
 1.5  2.25

julia> diff2_space
3×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
 3.0  3.0
```

See also: [`EvalDiff2!`](@ref)
"""
function EvalDiff2(max_degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    eval_space = similar(x, (max_degree + 1, length(x)))
    diff_space = similar(x, (max_degree + 1, length(x)))
    diff2_space = similar(x, (max_degree + 1, length(x)))
    EvalDiff2!(eval_space, diff_space, diff2_space, basis, x)
    eval_space, diff_space, diff2_space
end

struct ScaledBasis{B<:UnivariateBasis, V<:AbstractVector{<:Real}} <: UnivariateBasis
    basis::B
    scaling::V
end

function Evaluate!(space::AbstractMatrix{U}, basis::ScaledBasis, x::AbstractVector{U}) where {U}
    @argcheck size(space, 1) <= length(basis.scaling) DimensionMismatch
    Evaluate!(space, basis.basis, x)
    scaling = @view basis.scaling[1:size(space, 1)]
    @. space = space * scaling
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, basis::ScaledBasis, x::AbstractVector{U}) where {U}
    @argcheck size(eval_space, 1) <= length(basis.scaling) DimensionMismatch
    EvalDiff!(eval_space, diff_space, basis.basis, x)
    scaling = @view basis.scaling[1:size(eval_space, 1)]
    @. eval_space = eval_space * scaling
    @. diff_space = diff_space * scaling
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, diff2_space::AbstractMatrix{U}, basis::ScaledBasis, x::AbstractVector{U}) where {U}
    @argcheck size(eval_space, 1) <= length(basis.scaling) DimensionMismatch
    EvalDiff2!(eval_space, diff_space, diff2_space, basis.basis, x)
    scaling = @view basis.scaling[1:size(eval_space, 1)]
    @. eval_space = eval_space * scaling
    @. diff_space = diff_space * scaling
    @. diff2_space = diff2_space * scaling
end
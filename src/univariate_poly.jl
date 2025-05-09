export LegendrePolynomial, MonicLegendrePolynomial,
    ProbabilistHermitePolynomial, NormalizedProbabilistHermitePolynomial,
    PhysicistHermitePolynomial,
    JacobiPolynomial, MonicJacobiPolynomial,
    LaguerrePolynomial, ShiftedLegendrePolynomial

# Abstract type for univariate polynomials
abstract type Polynomial <: UnivariateBasis end

"""
Struct representing a _monic_ orthogonal polynomial family via three-term recurrence relation:

```math
p_{k+1} = (x - a_k)p_k - b_kp_{k-1}
```
"""
struct MonicOrthogonalPolynomial{Ak,Bk} <: Polynomial
    ak::Ak
    bk::Bk
end


"""
Struct representing an orthogonal polynomial family via three-term recurrence relation:

```math
L_kp_{k+1} = (m_kx - a_k)p_k - b_kp_{k-1}
```
"""
struct OrthogonalPolynomial{Lk,Mk,Ak,Bk,P0<:Number,P1<:Number,DP1<:Number} <: Polynomial
    lk::Lk
    mk::Mk
    ak::Ak
    bk::Bk
    phi0::P0
    phi1_offset::P1
    phi1_slope::DP1
    function OrthogonalPolynomial(lk::_Lk, mk::_Mk, ak::_Ak, bk::_Bk;
        phi0::_P0=true, phi1_offset::_P1=false, phi1_slope::_DP1=true
    ) where {_Lk,_Mk,_Ak,_Bk,_P0,_P1,_DP1}
        new{_Lk,_Mk,_Ak,_Bk,_P0,_P1,_DP1}(lk, mk, ak, bk, phi0, phi1_offset, phi1_slope)
    end
end

@inline _ProbHermiteAk(::Int)::Int = false
@inline _ProbHermiteBk(k::Int)::Int = k

# Probabilist Hermite polynomials `He_k`, orthogonal under `exp(-x^2/2)`
ProbabilistHermitePolynomial() = MonicOrthogonalPolynomial(_ProbHermiteAk, _ProbHermiteBk)

@inline _NormProbHermiteLk(k::Int) = sqrt(k+1)
@inline _NormProbHermiteMk = Returns(true)
@inline _NormProbHermiteAk = Returns(false)
@inline _NormProbHermiteBk(k::Int) = sqrt(k)
NormalizedProbabilistHermitePolynomial() = OrthogonalPolynomial(_NormProbHermiteLk, _NormProbHermiteMk, _NormProbHermiteAk, _NormProbHermiteBk)

@inline _PhysHermiteLk = Returns(true)
@inline _PhysHermiteMk = Returns(2)
@inline _PhysHermiteAk = Returns(false)
@inline _PhysHermiteBk(k::Int)::Int = 2k
const _PhysHermitePhi1_Slope = 2

# Physicist Hermite polynomials `H_k`, orthogonal under `exp(-x^2)`
PhysicistHermitePolynomial() = OrthogonalPolynomial(_PhysHermiteLk, _PhysHermiteMk, _PhysHermiteAk, _PhysHermiteBk, phi1_slope = _PhysHermitePhi1_Slope)

@inline _LegendreLk(k::Int)::Int = k + 1
@inline _LegendreMk(k::Int)::Int = 2k + 1
@inline _LegendreAk = Returns(false)
@inline _LegendreBk(k::Int)::Int = k
# Legendre polynomials `P_k`, orthogonal under `U[-1,1]`
LegendrePolynomial() = OrthogonalPolynomial(_LegendreLk, _LegendreMk, _LegendreAk, _LegendreBk)

@inline _MonicLegendreAk = Returns(false)
@inline _MonicLegendreBk(k::Int)::Float64 = (k * k) / muladd(4, k * k, -1.0)
# Monic Legendre polynomials `P_k`, orthogonal under `U[-1,1]`
MonicLegendrePolynomial() = MonicOrthogonalPolynomial(_MonicLegendreAk, _MonicLegendreBk)

@inline _ShiftedLegendreLk(k::Int)::Int = (k+1)
@inline _ShiftedLegendreMk(k::Int)::Int = muladd(4, k, 2)
@inline _ShiftedLegendreAk(k::Int)::Int = muladd(2, k, 1)
@inline _ShiftedLegendreBk(k::Int)::Int = k
const _ShiftedLegendrePhi1_Offset = -1
const _ShiftedLegendrePhi1_Slope = 2
ShiftedLegendrePolynomial() = OrthogonalPolynomial(_ShiftedLegendreLk, _ShiftedLegendreMk, _ShiftedLegendreAk, _ShiftedLegendreBk, phi1_offset = _ShiftedLegendrePhi1_Offset, phi1_slope = _ShiftedLegendrePhi1_Slope)

@inline _LaguerreLk(k::Int)::Int = k + 1
@inline _LaguerreMk(k::Int)::Int = -1
@inline _LaguerreAk(k::Int)::Int = -muladd(2, k, 1)
@inline _LaguerreBk(k::Int)::Int = k
const _LaguerrePhi1_Offset = true
const _LaguerrePhi1_Slope  = -1
LaguerrePolynomial() = OrthogonalPolynomial(_LaguerreLk, _LaguerreMk, _LaguerreAk, _LaguerreBk, phi1_offset = _LaguerrePhi1_Offset, phi1_slope = _LaguerrePhi1_Slope)

"""
    MonicJacobiPolynomial(α,β)

Monic Jacobi polynomials `P^(α,β)_k`, orthogonal on `[-1,1]` with weight `(1-x)^α(1+x)^β`
"""
function MonicJacobiPolynomial(alpha::T, beta::T) where {T}
    alpha == 0.0 && beta == 0.0 && return MonicLegendrePolynomial()

    @inline Ak(k::Int)::Float64 = (beta^2 - alpha^2) / ((alpha + beta + 2k) * (alpha + beta + 2k + 2))
    @inline Bk(k::Int)::Float64 = 4k * (k + alpha) * (k + beta) * (k + alpha + beta) / (((2k + alpha + beta)^2) * (2k + alpha + beta + 1) * (2k + alpha + beta - 1))
    MonicOrthogonalPolynomial(Ak, Bk)
end

"""
    JacobiPolynomial(α,β)

Jacobi polynomials `P^(α,β)_k`, orthogonal on `[-1,1]` with weight `(1-x)^α(1+x)^β`
"""
function JacobiPolynomial(alpha::T, beta::T) where {T}
    alpha == 0.0 && beta == 0.0 && return LegendrePolynomial()

    @inline Lk(k::Int)::T = 2(k + 1) * (k + 1 + alpha + beta) * (2k + alpha + beta)
    @inline Mk(k::Int)::T = (2k + alpha + beta + 1) * (2k + 2 + alpha + beta) * (2k + alpha + beta)
    @inline Ak(k::Int)::T = (beta^2 - alpha^2) * (2k + alpha + beta + 1)
    @inline Bk(k::Int)::T = 2 * (k + alpha) * (k + beta) * (2k + alpha + beta + 2)
    phi1_offset = (alpha-beta)/2
    phi1_slope  = (alpha+beta+2)/2
    OrthogonalPolynomial(Lk, Mk, Ak, Bk; phi1_offset, phi1_slope)
end

@inline function getPhi0(::Type{U},::MonicOrthogonalPolynomial) where {U}
    one(U)
end

@inline function getPhi1Slope(::Type{U},::MonicOrthogonalPolynomial) where {U}
    one(U)
end

@inline function getPhi1Offset(::Type{U},p::MonicOrthogonalPolynomial) where {U}
    -convert(U,p.ak(0))
end

@inline function getPhi0(::Type{U}, p::OrthogonalPolynomial) where {U}
    convert(U,p.phi0)
end

@inline function getPhi1Slope(::Type{U}, p::OrthogonalPolynomial) where {U}
    convert(U,p.phi1_slope)
end

@inline function getPhi1Offset(::Type{U}, p::OrthogonalPolynomial) where {U}
    convert(U,p.phi1_offset)
end

function Evaluate!(space::AbstractMatrix{U}, poly::OrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    N, deg = length(x), size(space, 1) - 1
    @argcheck size(space, 2) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_off, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        space[0+1, :] .= p0
        return
    end

    (; lk, mk, ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            space[0+1, j] = p0
            space[1+1, j] = muladd(p1_slope, x[j], p1_off)
            @simd for k in 1:deg-1
                idx = k + 1
                space[idx+1, j] = muladd(muladd(mk(k), x[j], -ak(k)), space[idx, j], -bk(k) * space[idx-1, j])
                space[idx+1, j] /= lk(k)
            end
            nothing
        end
    end
    nothing
end

function EvaluateDegree!(space::AbstractVector{U}, deg::Int, poly::OrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    N = length(x)
    @argcheck length(space) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_off, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)

    if deg == 0
        space .= p0
        return
    end

    (; lk, mk, ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            old_val = p0
            space[j] = muladd(p1_slope, x[j], p1_off)
            @simd for k in 1:deg-1
                tmp = space[j]
                space[j] = muladd(muladd(mk(k), x[j], -ak(k)), space[j], -bk(k) * old_val)
                space[j] /= lk(k)
                old_val = tmp
            end
        end
    end
    nothing
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, poly::OrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    N, deg = length(x), size(eval_space, 1) - 1
    @argcheck size(eval_space, 2) == N DimensionMismatch
    @argcheck size(diff_space, 1) == deg + 1 && size(diff_space, 2) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_off, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        eval_space[0+1, :] .= p0
        diff_space[0+1, :] .= zero(U)
        return
    end
    (; lk, mk, ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            eval_space[0+1, j] = p0
            diff_space[0+1, j] = zero(U)
            eval_space[1+1, j] = muladd(p1_slope, x[j], p1_off)
            diff_space[1+1, j] = p1_slope
            @simd for k in 1:deg-1
                idx = k + 1
                pk_ = muladd(mk(k), x[j], -ak(k))
                eval_space[idx+1, j] = muladd(pk_, eval_space[idx, j], -bk(k) * eval_space[idx-1, j])
                diff_space[idx+1, j] = muladd(mk(k), eval_space[idx, j], muladd(pk_, diff_space[idx, j], -bk(k) * diff_space[idx-1, j]))
                eval_space[idx+1, j] /= lk(k)
                diff_space[idx+1, j] /= lk(k)
            end
        end
    end
    nothing
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, diff2_space::AbstractMatrix{U}, poly::OrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    N, deg = length(x), size(eval_space, 1) - 1

    @argcheck size(eval_space, 2) == N DimensionMismatch
    @argcheck size(diff_space, 1) == deg + 1 && size(diff_space, 2) == N DimensionMismatch
    @argcheck size(diff2_space, 1) == deg + 1 && size(diff2_space, 2) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_off, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        eval_space[0+1, :] .= p0
        diff_space[0+1, :] .= zero(U)
        diff2_space[0+1, :] .= zero(U)
        return
    end

    (; lk, mk, ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            eval_space[0+1, j] = p0
            diff_space[0+1, j] = zero(U)
            diff2_space[0+1, j] = zero(U)
            eval_space[1+1, j] = muladd(p1_slope, x[j], p1_off)
            diff_space[1+1, j] = p1_slope
            diff2_space[1+1, j] = zero(U)

            @simd for k in 1:deg-1
                idx = k + 1
                @muladd pk_ = mk(k) * x[j] - ak(k)
                @muladd eval_space[idx+1, j] = pk_ * eval_space[idx, j] - bk(k) * eval_space[idx-1, j]
                diff_space[idx+1, j] = muladd(mk(k), eval_space[idx, j], muladd(pk_, diff_space[idx, j], -bk(k) * diff_space[idx-1, j]))
                diff2_space[idx+1, j] = muladd(2mk(k), diff_space[idx, j], muladd(pk_, diff2_space[idx, j], -bk(k) * diff2_space[idx-1, j]))

                eval_space[idx+1, j] /= lk(k)
                diff_space[idx+1, j] /= lk(k)
                diff2_space[idx+1, j] /= lk(k)
            end
        end
    end
    nothing
end

function Evaluate!(space::AbstractMatrix{U}, poly::MonicOrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    # Evaluate a monic orthogonal polynomial; may be faster than the general case
    N, deg = length(x), size(space, 1) - 1
    @argcheck size(space, 2) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_off, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        space[0+1, :] .= p0
        return
    end

    (; ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            space[0+1, j] = p0
            space[1+1, j] = muladd(p1_slope, x[j], p1_off)
            @simd for k in 1:deg-1
                idx = k + 1
                space[idx+1, j] = muladd(x[j] - ak(k), space[idx, j], -bk(k) * space[idx-1, j])
            end
        end
    end
    nothing
end

function EvaluateDegree!(space::AbstractVector{U}, deg::Int, poly::MonicOrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    # Evaluate a monic orthogonal polynomial; may be faster than the general case
    N = length(x)
    @argcheck length(space) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_offset, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        space .= p0
        return
    end

    (; ak, bk) = poly
    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            old_val = p0
            space[j] = muladd(p1_slope, x[j], p1_offset)
            @simd for k in 1:deg-1
                tmp = space[j]
                space[j] = muladd(x[j] - ak(k), space[j], -bk(k) * old_val)
                old_val = tmp
            end
        end
    end
    nothing
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, poly::MonicOrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    # Evaluate a monic orthogonal polynomial and its derivative; may be faster than the general case
    N, deg = length(x), size(eval_space, 1) - 1
    @argcheck size(eval_space, 2) == N DimensionMismatch
    @argcheck size(diff_space, 1) == deg + 1 && size(diff_space, 2) == N DimensionMismatch
    @argcheck deg >= 0
    p0, p1_offset, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        eval_space[0+1, :] .= p0
        diff_space[0+1, :] .= zero(U)
        return
    end

    (; ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            eval_space[0+1, j] = p0
            diff_space[0+1, j] = zero(U)

            eval_space[1+1, j] = muladd(p1_slope, x[j], p1_offset)
            diff_space[1+1, j] = p1_slope

            @simd for k in 1:deg-1
                idx = k + 1
                x_sub_a = x[j] - ak(k)
                eval_space[idx+1, j] = muladd(x_sub_a, eval_space[idx, j], -bk(k) * eval_space[idx-1, j])
                diff_space[idx+1, j] = muladd(x_sub_a, diff_space[idx, j], muladd(-bk(k), diff_space[idx-1, j], eval_space[idx, j]))
            end
        end
    end
    nothing
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, diff2_space::AbstractMatrix{U}, poly::MonicOrthogonalPolynomial, x::AbstractVector{U}; kwargs...) where {U}
    N, deg = length(x), size(eval_space, 1) - 1
    @argcheck size(eval_space, 2) == N DimensionMismatch
    @argcheck size(diff_space, 1) == deg + 1 && size(diff_space, 2) == N DimensionMismatch
    @argcheck size(diff2_space, 1) == deg + 1 && size(diff2_space, 2) == N DimensionMismatch
    @argcheck deg >= 0

    p0, p1_offset, p1_slope = getPhi0(U, poly), getPhi1Offset(U, poly), getPhi1Slope(U, poly)
    if deg == 0
        eval_space[0+1, :] .= p0
        diff_space[0+1, :] .= zero(U)
        diff2_space[0+1, :] .= zero(U)
        return
    end

    (; ak, bk) = poly

    AK.foreachindex(x; kwargs...) do j
        @inbounds begin
            eval_space[0+1, j] = p0
            diff_space[0+1, j] = zero(U)
            diff2_space[0+1, j] = zero(U)

            eval_space[1+1, j] = muladd(p1_slope, x[j], p1_offset)
            diff_space[1+1, j] = p1_slope
            diff2_space[1+1, j] = zero(U)

            @simd for k in 1:deg-1
                idx = k + 1
                x_sub_a = x[j] - ak(k)
                eval_space[idx+1, j] = muladd(x_sub_a, eval_space[idx, j], -bk(k) * eval_space[idx-1, j])
                diff_space[idx+1, j] = muladd(x_sub_a, diff_space[idx, j], muladd(-bk(k), diff_space[idx-1, j], eval_space[idx, j]))
                diff2_space[idx+1, j] = muladd(x_sub_a, diff2_space[idx, j], muladd(-bk(k), diff2_space[idx-1, j], 2 * diff_space[idx, j]))
            end
        end
    end
    nothing
end
export gausslaguerre_lobatto
export clenshawcurtis01, clenshawcurtis01_nested
export gausspatterson01, gausspatterson01_nested
export leja01_closed_nested, leja01_open_nested

function coeff_diff_laguerre(degree,k,is_monic=false)
	@assert !(k < 0 || k > degree) "invalid k = $k, degree = $degree"
	k == 0 && return 0
	if is_monic
		sgn = (-1)^isodd(degree-k)
		k == degree && return 1
		return ((binomial(degree-1,k-1)^2*factorial(degree-k)*degree)÷k)*sgn
	end
	sgn = (-1)^isodd(k)
	ret = sgn/factorial(k)
	k == degree && return ret
	return ret*binomial(degree-1,k-1)
end

function CompanionDiffLaguerre(degree)
	coeffs = coeff_diff_laguerre.((degree,), 0:degree, true)
	A = Int[-coeffs[end-1:-1:2]';I(degree-2) zeros(Int,degree-2)]
	A
end

"""
	gausslaguerre_lobatto(n)
Create an `n`-point quadrature rule to estimate the integral

```math
\\int_0^\\infty f(t)\\exp(-t) dt
```

This method guarantees the first evaluation will be at `t_1=0` and is exact
on polynomials up to degree `2n-2` (one degree of freedom is lost to fixing `t_1`).
"""
function gausslaguerre_lobatto(n)
	@assert n > 0 && n < 20 "Expected 0 < n < 20, got n = $n"
	n == 1 && return [0.],[1.]
	A = CompanionDiffLaguerre(n)
	pts = [0;eigvals(A)]
	poly = LaguerrePolynomial()
	evals = EvaluateDegree(n-1, poly, pts)
	for k in eachindex(evals)
		evals[k] = 1/(n*evals[k]^2)
	end
	pts, evals
end

# Inspired by implementation in ChaosPy: https://github.com/jonathf/chaospy/blob/53000bbb04f8d3f9908ebbf1be6bf139a21c2e6e/chaospy/quadrature/clenshaw_curtis.py#L76
function clenshawcurtis01(order::Int)
    if order == 0
        return [0.5], [1.0]
    elseif order == 1
        return [0.0, 1.0], [0.5, 0.5]
    end

    theta = (order .- (0:order)) .* π / order
    abscissas = 0.5 .* cos.(theta) .+ 0.5

    steps = 1:2:(order - 1)
    L = length(steps)
    remains = order - L

    beta = vcat(2.0 ./ (steps .* (steps .- 2)), [1.0 / steps[end]], zeros(remains))
    beta = -beta[1:(end - 1)] .- reverse(beta[2:end])

    gamma = -ones(order)
    gamma[L + 1] += order
    gamma[remains + 1] += order
    gamma ./= order^2 - 1 + (order % 2)

    weights = rfft(beta + gamma) / order
    @assert maximum(imag.(weights)) < 1e-15
    weights = real.(weights)
    weights = vcat(weights, reverse(weights)[(2 - (order % 2)):end]) ./ 2

    return abscissas, weights
end
"""
    CreateQuadratureWeights(pts, functions, true_integrals)
Simple method to create quadrature weights to integrate functions correctly.

# Arguments
- `pts::AbstractVector`-- vector of points
- `functions(N, pt)`-- A function to evaluate the first `N` functions at 1d `pt`
- `true_integrals(N)`-- A function to return the true integrals of rhe first `N` functions
- `verbose::Bool`-- whether to be verbose
"""
function CreateQuadratureWeights(pts::AbstractVector, functions, true_integrals, verbose=false)
    N = length(pts)
    function_evals = functions(N, pts)
    N == 1 && return true_integrals(1)/function_evals[]
    function_ints = true_integrals(N)
    verbose && @info "Conditioning: $(cond(function_evals))"
    function_evals\function_ints
end

__GAUSSPATTERSON = open(deserialize, joinpath(@__DIR__,"serial","gausspatterson.ser"), "r")
__UNIFORMLEJA_CLOSED = open(deserialize, joinpath(@__DIR__, "serial", "uniformleja_closed.ser"), "r")
__UNIFORMLEJA_OPEN   = open(deserialize, joinpath(@__DIR__, "serial", "uniformleja_open.ser"), "r")

"""
    gausspatterson01(n)
Nested gausspatterson rule to integrate over [0,1], adapted from John Burkardt's implementation.

n must be 1, 3, 7, 15, 31, 63, 127, 255 or 511. Exact for polynomials up to degree `3*2^index - 1` for index > 1.

See [here](https://people.math.sc.edu/Burkardt/m_src/patterson_rule/patterson_rule.html) for more information
"""
function gausspatterson01(n::Int)
    j = floor(Int, log2(n+1))
    @assert n == 2^j - 1 "n=$n, expected n in [1, 3, 7, 15, 31, 63, 127, 255 or 511]"
    __GAUSSPATTERSON[j]
end

clenshawcurtis01_nested(j) = clenshawcurtis01(j == 0 ? 0 : 2^j)
gausspatterson01_nested(j) = gausspatterson01(2^(j+1) - 1)

function create01quad(pts)
    leg_poly = LegendrePolynomial()
    leg_poly_eval = (N,x) -> Evaluate(N-1, leg_poly, x)
    function TrueIntegralLegendre(n)
        n == 1 && return [1.0]
        ret = zeros(n)
        ret[1] = 1.0
        ret
    end
    wts = CreateQuadratureWeights(pts, leg_poly_eval, TrueIntegralLegendre)
    # Adjust pts for domain
    @. pts = (pts + 1) / 2
    pts, wts
end

function leja01_closed(n)
    n > length(__UNIFORMLEJA_CLOSED) && throw(ArgumentError("Expected n < $(length(__UNIFORMLEJA_CLOSED)), got n=$n"))
    pts = __UNIFORMLEJA_CLOSED[1:n]
    create01quad(pts)
end

function leja01_open(n)
    n > length(__UNIFORMLEJA_OPEN) && throw(ArgumentError("Expected n < $(length(__UNIFORMLEJA_OPEN)), got n=$n"))
    pts = __UNIFORMLEJA_OPEN[1:n]
    create01quad(pts)
end

# Skips two-point rule, which is identical to level 0 due to weight construction
leja01_closed_nested(level) = leja01_closed(level + 1 + (level > 0))
leja01_open_nested(level)   = leja01_open(level + 1 + (level > 0))

# Make it possible to iterate over feasible values of n
using Base: eachindex
Base.eachindex(::typeof(leja01_closed_nested)) = 1:(length(__UNIFORMLEJA_CLOSED) - 3)
Base.eachindex(::typeof(leja01_open_nested)) = 1:(length(__UNIFORMLEJA_OPEN) - 3)
Base.eachindex(::typeof(gausspatterson01_nested)) = 1:(length(__GAUSSPATTERSON) - 1)
Base.eachindex(::typeof(gausspatterson01)) = [1, 3, 7, 15, 31, 63, 127, 255, 511]

"""
    exactness(rule, n)
Return the exactness of the quadrature rule `rule` for argument `n`.

Some rules, e.g., Clenshaw-Curtis, have "approximate" exactness beyond the analytical exactness.
"""
exactness

exactness(::typeof(leja01_closed_nested), n::Int) = n + 1
exactness(::typeof(leja01_open_nested), n::Int) = n + 1
exactness(::typeof(gausspatterson01_nested), n::Int) = 3*2^(n) - 1
exactness(::typeof(gausspatterson01), n::Int) = 3*(n+1)÷2 - 1
exactness(::typeof(clenshawcurtis01), n::Int) = n-1
exactness(::typeof(clenshawcurtis01_nested), n::Int) = 2^n - 1
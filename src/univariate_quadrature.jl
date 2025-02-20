export gausslaguerre_lobatto

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
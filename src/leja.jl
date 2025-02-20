module FindLeja
# This module is included for reproducibility purposes. However, to minimize dependencies, we use a serialized version of the points
# and create the weights on the fly
# The default optimization functions require the dependencies: Optimization, OptimizationOptimJL, ForwardDiff
# TODO: Add this as a package extension

# True inverse christoffel weighting for nth leja point on U(-1,1)
function LogChristoffelTrue(x, n, normalize=true)
	evals = EvaluateLegendrePolynomial(n - 1, x)
	christoffel = 0.0
	if normalize
		christoffel = sum(evals[j]^2 / LegendreSquareNorm(j - 1) for j in eachindex(evals))
	else
		christoffel = sum(x -> x^2, evals)
	end
	-0.5log(christoffel)
end

# Given previous leja points and density function, this is the evaluation
# of the optimization loss. Takes offset of points into account when we want
# to avoid certain points (e.g., (-1,1) vs [-1,1])
function LejaLoss(x, p)
	logdensity, prev_leja, offset = p
	log_diffs = sum(L -> log(abs(L - x[])), prev_leja)
	log_inv_sqrt_christoffel = logdensity(x[], length(prev_leja)-offset)
	-(log_diffs + log_inv_sqrt_christoffel)
end

# Simple wrapper for optimization function
function ExampleLejaOptimizationSetup_ForwardDiff(fcn)
    OptimizationFunction(fcn, Optimization.AutoForwardDiff())
end

# Simple wrapper for performing example optimization
function ExampleLejaConstrainedOptimizer_OptimizationOptimJL(fcn, u0, params, lb, ub)
	# Problem setup
	prob = OptimizationProblem(fcn, [u0], params, lb=[lb], ub=[ub])
	reltol = 1000eps()
	sol = solve(prob, LBFGS(); reltol)
	bound_tol = 10reltol

	# If we find a minimum inside the interval, treat as "trust region"
	if sol.u[] > lb + bound_tol && sol.u[] < ub - bound_tol
		local_prob = OptimizationProblem(fcn, sol.u, params)
		sol_local = solve(local_prob, Newton(), reltol=100eps())
		# Ensure the trust region solver actually found something in the domain
		if sol_local.u[] > lb && sol_local.u[] < ub
			sol = sol_local
		end
	end
	sol.u[], sol.objective[]
end

function FindLejaPoints1d(N, prefix_points=Float64[];
	constrained_minimizer=ExampleLejaConstrainedOptimizer_OptimizationOptimJL,
	minimization_setup=ExampleLejaOptimizationSetup_ForwardDiff,
	first_leja_pt=0.,
	leja_logdensity=LogChristoffelTrue)

	leja_pts = zeros(N)
	offset = length(prefix_points)
	leja_pts[1:offset] .= prefix_points
	leja_pts[offset+1] = first_leja_pt
	N -= offset+1

	loss_fcn = minimization_setup(LejaLoss)

    # Suggested: use a progress bar for this loop
	for j in 1:N
		leja_pts_sorted = sort(@view(leja_pts[1:offset+j]))
		N_j = length(leja_pts_sorted)
		optim_j, optim_j_obj, optim_j_lb, optim_j_ub = 0.0, Inf, -Inf, Inf
		# Check between all the poles in the process
		for k in 1:N_j+1
			# Create the box bounding the poles
			lb = k == 1 ? -1.0 : leja_pts_sorted[k-1]
			ub = k > N_j ? 1.0 : leja_pts_sorted[k]

			# Check that we aren't starting too close to the domain's edge
			(ub-lb) < 100eps() && continue

			u0 = (ub + lb) / 2

			# Find the best point using given minimizer
			params = (leja_logdensity, leja_pts_sorted, offset)
			u_jk, objective_jk = constrained_minimizer(loss_fcn, u0, params, lb, ub)

			# If this u0 gives a better minimizer, use it
			if objective_jk < optim_j_obj
				optim_j = u_jk
				optim_j_obj = objective_jk
				optim_j_lb = lb
				optim_j_ub = ub
			end
		end
		@info "" j optim_j optim_j_obj optim_j_lb optim_j_ub
		leja_pts[offset+j+1] = optim_j
	end
	leja_pts[offset+1:end]
end

end # module FindLeja
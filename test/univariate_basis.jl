p, N_pts = 6, 100
rng = Xoshiro(284028)
pts = 2rand(rng, N_pts) .- 1
basis1 = MonicJacobiPolynomial(0.5, 0.75)
basis2 = JacobiPolynomial(0.5, 0.75)

@testset "Univariate basis evaluation" begin
    for (j,basis) in enumerate([basis1, basis2])
        eval_space_ref = Matrix{Float64}(undef, p+1, N_pts)
        diff_space_ref = similar(eval_space_ref)
        diff2_space_ref = similar(eval_space_ref)
        EvalDiff2!(eval_space_ref, diff_space_ref, diff2_space_ref, basis, pts)

        @testset "Out-of-place evaluation $j" begin
            eval_space = Evaluate(p, basis, pts)
            @test eval_space ≈ eval_space_ref atol=1e-14

            eval_space, diff_space = EvalDiff(p, basis, pts)
            @test eval_space ≈ eval_space_ref atol=1e-14
            @test diff_space ≈ diff_space_ref atol=1e-14

            eval_space, diff_space, diff2_space = EvalDiff2(p, basis, pts)
            @test eval_space ≈ eval_space_ref atol=1e-14
            @test diff_space ≈ diff_space_ref atol=1e-14
            @test diff2_space ≈ diff2_space_ref atol=1e-14
        end
    end
end

@testset "ScaledBasis tests" begin
    polys = ProbabilistHermitePolynomial()
    inv_sqrt_factorial = k -> k == 0 ? 1 : exp(-0.5sum(log, 1:k))
    scaling = inv_sqrt_factorial.(0:p)
    norm_polys = ScaledBasis(polys, scaling)
    eval_true, diff_true, diff2_true = EvalDiff2(p, polys, pts)
    eval_true .*= scaling
    diff_true .*= scaling
    diff2_true .*= scaling
    eval_space = Evaluate(p, norm_polys, pts)
    @test eval_space ≈ eval_true atol=1e-14
    eval_space, diff_space = EvalDiff(p, norm_polys, pts)
    @test eval_space ≈ eval_true atol=1e-14
    @test diff_space ≈ diff_true atol=1e-14
    eval_space, diff_space, diff2_space = EvalDiff2(p, norm_polys, pts)
    @test eval_space ≈ eval_true atol=1e-14
    @test diff_space ≈ diff_true atol=1e-14
    @test diff2_space ≈ diff2_true atol=1e-14
end
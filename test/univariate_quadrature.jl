@testset "all univariate quadrature" begin
    @testset "Gauss-Laguerre-Lobatto quadrature" begin
        for n in 1:18
            pts, wts = gausslaguerre_lobatto(n)
            evals = ones(n)
            for degree in 0:2n-2
                monomial_int = evals'wts
                exact_int = factorial(big(degree))
                # Manually set relative tolerance as larger for high-degree polynomials due to numerical issues
                rtol = degree < 13 && n < 13 ? 5e-12 : 1e-8
                @test monomial_int ≈ exact_int rtol=rtol
                evals .*= pts
            end
        end
    end

    using Base: eachindex
    # Only test first 10 quadrature rules for nested CC
    Base.eachindex(::typeof(clenshawcurtis01)) = 1:64
    Base.eachindex(::typeof(clenshawcurtis01_nested)) = 1:10

    @testset "Quadrature [0,1]" begin
        for qrule in [leja01_open_nested, leja01_closed_nested, gausspatterson01_nested, clenshawcurtis01, clenshawcurtis01_nested]
            prev_pts = []
            for n in eachindex(qrule)
                pts, wts = qrule(n)
                qrule != clenshawcurtis01 && (@test all(x in pts for x in prev_pts))
                prev_pts = copy(pts)
                evals = ones(length(pts))
                pts_n_res = true
                for degree in 0:UnivariateApprox.exactness(qrule, n)
                    monomial_int = evals'wts
                    # Exact integral of monomial x^degree on [0,1] is 1/(degree + 1)
                    exact_int = 1.0 / (degree + 1)
                    # Manually set relative tolerance as larger for high-degree polynomials due to numerical issues
                    rtol = degree < 13 && n < 13 ? 5e-12 : 1e-8
                    res_degree = isapprox(monomial_int, exact_int; rtol)
                    if !res_degree
                        @info "" qrule n degree monomial_int exact_int rtol
                    end
                    pts_n_res &= res_degree
                    evals .*= pts
                end
                @test pts_n_res
            end
        end
    end
end
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
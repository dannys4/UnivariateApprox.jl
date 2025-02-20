for n in 1:18
    pts, wts = gausslaguerre_lobatto(n)
    evals = ones(n)
    for degree in 0:2n-2
        monomial_int = evals'wts
        exact_int = factorial(big(degree))
        rtol = degree < 13 && n < 13 ? 1e-12 : 1e-8
        if !isapprox(monomial_int, exact_int, rtol=rtol)
            @info "" degree n abs(monomial_int - exact_int)/abs(exact_int)
        end
        @test monomial_int ≈ exact_int rtol=rtol
        evals .*= pts
    end
end
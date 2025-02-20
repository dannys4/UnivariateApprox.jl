using UnivariateApprox
using UnivariateApprox: MonicOrthogonalPolynomial
using Test, Random

Monomials() = MonicOrthogonalPolynomial(Returns(0.),Returns(0.))

@testset "UnivariateApprox.jl" begin
    # Test quadrature rules
    include("gausslaguerre_lobatto.jl")

    # Test basis functions
    include("univariate_basis.jl")
    include("univariate_poly.jl")
    include("mollified_basis.jl")
end

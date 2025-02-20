module UnivariateApprox
using MuladdMacro, LinearAlgebra

# Quadrature rules
include("gausslaguerre_lobatto.jl")

# Basis functions
include("univariate_basis.jl")
include("univariate_poly.jl")
include("mollified_basis.jl")

end

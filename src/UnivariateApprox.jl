module UnivariateApprox
using MuladdMacro, LinearAlgebra, Serialization, FFTW

# Quadrature rules
include("univariate_quadrature.jl")

# Basis functions
include("univariate_basis.jl")
include("univariate_poly.jl")
include("mollified_basis.jl")

end

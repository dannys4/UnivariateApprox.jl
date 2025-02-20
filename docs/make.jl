using UnivariateApprox
using Documenter

DocMeta.setdocmeta!(UnivariateApprox, :DocTestSetup, :(using UnivariateApprox); recursive=true)

makedocs(;
    modules=[UnivariateApprox],
    authors="Daniel Sharp <dannys4@mit.edu> and contributors",
    sitename="UnivariateApprox.jl",
    format=Documenter.HTML(;
        canonical="https://dannys4.github.io/UnivariateApprox.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dannys4/UnivariateApprox.jl",
    devbranch="main",
)

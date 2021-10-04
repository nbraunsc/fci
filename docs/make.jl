using fci
using Documenter

DocMeta.setdocmeta!(fci, :DocTestSetup, :(using fci); recursive=true)

makedocs(;
    modules=[fci],
    authors="Nicole Braunscheidel, Virginia Tech",
    repo="https://github.com/nbraunsc/fci.jl/blob/{commit}{path}#{line}",
    sitename="fci.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nbraunsc.github.io/fci.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nbraunsc/fci.jl",
)

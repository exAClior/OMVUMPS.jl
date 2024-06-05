using OMVUMPS
using Documenter

DocMeta.setdocmeta!(OMVUMPS, :DocTestSetup, :(using OMVUMPS); recursive=true)

makedocs(;
    modules=[OMVUMPS],
    authors="Yusheng Zhao <yushengzhao2020@outlook.com> and contributors",
    sitename="OMVUMPS.jl",
    format=Documenter.HTML(;
        canonical="https://exAClior.github.io/OMVUMPS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/exAClior/OMVUMPS.jl",
    devbranch="main",
)

using Documenter, TrueSkill

makedocs(
    modules = [TrueSkill],
    format = Documenter.HTML(),
    checkdocs = :exports,
    sitename = "TrueSkill.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/glandfried/TrueSkill.jl.git",
)

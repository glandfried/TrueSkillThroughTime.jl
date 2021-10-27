using Documenter, Revise
using TrueSkillThroughTime
deploydocs(repo = "github.com/glandfried/TrueSkillThroughTime.jl.git")
makedocs(
    modules = [TrueSkillThroughTime],
    format = Documenter.HTML(),
    checkdocs = :exports,
    sitename = "TrueSkillThroughTime.jl",
    pages = Any["index.md"]
)


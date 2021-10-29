using Documenter
#using Revise
using TrueSkillThroughTime
makedocs(
    modules = [TrueSkillThroughTime],
    format = Documenter.HTML(),
    checkdocs = :exports,
    sitename = "TrueSkillThroughTime.jl",
    pages = Any["index.md"]
)
deploydocs(repo = "github.com/glandfried/TrueSkillThroughTime.jl")


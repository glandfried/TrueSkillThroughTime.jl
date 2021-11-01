using Documenter
#using Revise
using TrueSkillThroughTime
makedocs(
    modules = [TrueSkillThroughTime],
    format = Documenter.HTML(),
    checkdocs = :exports,
    sitename = "TrueSkillThroughTime.jl",
    pages = [
        "index.md",
        "Sections" => ["man/causal.md", "man/gaussian.md", "man/player.md", "man/game.md", "man/history.md", "man/examples.md"]
    ]
)
#deploydocs(
#    repo = "github.com/glandfried/TrueSkillThroughTime.jl",
#    forcepush = true
#)


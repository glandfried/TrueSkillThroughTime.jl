using CSV
using Dates
include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using DataFrames

data = CSV.read("input/history.csv", DataFrame)
days = Dates.value.(data[:,"time_start"] .- Date("1900-01-01"))
#dates = Date.(Dates.UTD.(days .+ Dates.value(Date("1900-01-01"))))

composition = [ r.double == "t" ? [[r.w1_id,r.w1_id*r.ground,r.w2_id,r.w2_id*r.ground],[r.l1_id,r.l1_id*r.ground,r.l2_id,r.l2_id*r.ground]] : [[r.w1_id,r.w1_id*r.ground],[r.l1_id,r.l1_id*r.ground]] for r in eachrow(data) ]   

players = Set(vcat(([ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]...)...))

priors = Dict([(p, ttt.Player(ttt.Gaussian(0., 1.6), 1.0, 0.036) ) for p in players])

function fit()
    h = ttt.History(composition=composition, times = days, sigma = 1.0, gamma = 0.01, beta = 0.0, priors = priors)
    ttt.convergence(h,epsilon=0.01, iterations=10)
    return h
end
@time h = fit()
ttt.log_evidence(h)
lc = ttt.learning_curves(h)

#exp(-265780/ 447028)
exp(-273383/ 447028)


federer = "f324"
nadal = "n409"
djokovic = "d643"
vilas = "v028"

dict = Dict{String,Vector}()
dict["federer_mu"] = [tp[2].mu for tp in lc["f324"]]
dict["federer_sigma"]= [tp[2].sigma for tp in lc["f324"]]
dict["federer_time"] = [tp[1] for tp in lc["f324"]]
dict["federer_mu_hard"] = [tp[2].mu for tp in lc["f324Hard"]]
dict["federer_sigma_hard"]= [tp[2].sigma for tp in lc["f324Hard"]]
dict["federer_time_hard"]= [tp[1] for tp in lc["f324Hard"]]
dict["federer_mu_clay"] = [tp[2].mu for tp in lc["f324Clay"]]
dict["federer_sigma_clay"]= [tp[2].sigma for tp in lc["f324Clay"]]
dict["federer_time_clay"]= [tp[1] for tp in lc["f324Clay"]]
dict["federer_mu_carpet"] = [tp[2].mu for tp in lc["f324Carpet"]]
dict["federer_sigma_carpet"]= [tp[2].sigma for tp in lc["f324Carpet"]]
dict["federer_time_carpet"]= [tp[1] for tp in lc["f324Carpet"]]
dict["federer_mu_grass"] = [tp[2].mu for tp in lc["f324Grass"]]
dict["federer_sigma_grass"]= [tp[2].sigma for tp in lc["f324Grass"]]
dict["federer_time_grass"]= [tp[1] for tp in lc["f324Grass"]]

dict["nadal_mu"] = [tp[2].mu for tp in lc["n409"]]
dict["nadal_sigma"] = [tp[2].sigma for tp in lc["n409"]]
dict["nadal_time"] = [tp[1] for tp in lc["n409"]]
dict["nadal_mu_hard"] = [tp[2].mu for tp in lc["n409Hard"]]
dict["nadal_sigma_hard"] = [tp[2].sigma for tp in lc["n409Hard"]]
dict["nadal_time_hard"] = [tp[1] for tp in lc["n409Hard"]]
dict["nadal_mu_clay"] = [tp[2].mu for tp in lc["n409Clay"]]
dict["nadal_sigma_clay"] = [tp[2].sigma for tp in lc["n409Clay"]]
dict["nadal_time_clay"] = [tp[1] for tp in lc["n409Clay"]]
dict["nadal_mu_carpet"] = [tp[2].mu for tp in lc["n409Carpet"]]
dict["nadal_sigma_carpet"] = [tp[2].sigma for tp in lc["n409Carpet"]]
dict["nadal_time_carpet"] = [tp[1] for tp in lc["n409Carpet"]]
dict["nadal_mu_grass"] = [tp[2].mu for tp in lc["n409Grass"]]
dict["nadal_sigma_grass"] = [tp[2].sigma for tp in lc["n409Grass"]]
dict["nadal_time_grass"] = [tp[1] for tp in lc["n409Grass"]]

dict["djokovic_mu"] = [tp[2].mu for tp in lc["d643"]]
dict["djokovic_sigma"] = [tp[2].sigma for tp in lc["d643"]]
dict["djokovic_time"] = [tp[1] for tp in lc["d643"]]
dict["djokovic_mu_hard"] = [tp[2].mu for tp in lc["d643Hard"]]
dict["djokovic_sigma_hard"] = [tp[2].sigma for tp in lc["d643Hard"]]
dict["djokovic_time_hard"] = [tp[1] for tp in lc["d643Hard"]]
dict["djokovic_mu_clay"] = [tp[2].mu for tp in lc["d643Clay"]]
dict["djokovic_sigma_clay"] = [tp[2].sigma for tp in lc["d643Clay"]]
dict["djokovic_time_clay"] = [tp[1] for tp in lc["d643Clay"]]
dict["djokovic_mu_carpet"] = [tp[2].mu for tp in lc["d643Carpet"]]
dict["djokovic_sigma_carpet"] = [tp[2].sigma for tp in lc["d643Carpet"]]
dict["djokovic_time_carpet"] = [tp[1] for tp in lc["d643Carpet"]]
dict["djokovic_mu_grass"] = [tp[2].mu for tp in lc["d643Grass"]]
dict["djokovic_sigma_grass"] = [tp[2].sigma for tp in lc["d643Grass"]]
dict["djokovic_time_grass"] = [tp[1] for tp in lc["d643Grass"]]

maxlen = maximum([length(value) for (key, value) in dict])

using DataFrames
df = DataFrame(Dict(key => [value;repeat([missing],maxlen-length(value))] for (key, value) in dict))

CSV.write("output/atp_ground_learning_curves.csv", df; header=true)

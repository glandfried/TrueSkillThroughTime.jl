using CSV
using Dates
include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill

data = CSV.read("input/history.csv")
times = Dates.value.(data[:,"time_start"] .- Date("1900-1-1")) .- data[:,"round_number"]
composition = [ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]   
fit function()
    h = ttt.History(composition=composition, times = times, sigma = 1.6, gamma = 0.036)
    ttt.convergence(h,epsilon=0.01, iterations=10)
    return h
end
@time h = fit()
lc = ttt.learning_curves(h)


federer = "f324"
nadal = "n409"
djokovic = "d643"
sampras = "s402"
aggasi = "a092"
vilas = "v028"
bjorn_borg = "b058"
john_mcenroe = "m047"

dict = Dict{String,Vector}()
dict["federer_mu"] = [tp[2].mu for tp in lc["f324"]]
dict["federer_sigma"]= [tp[2].sigma for tp in lc["f324"]]
dict["federer_time"] = [tp[1] for tp in lc["f324"]]
dict["nadal_mu"] = [tp[2].mu for tp in lc["n409"]]
dict["nadal_sigma"] = [tp[2].sigma for tp in lc["n409"]]
dict["nadal_time"] = [tp[1] for tp in lc["n409"]]
dict["djokovic_mu"] = [tp[2].mu for tp in lc["d643"]]
dict["djokovic_sigma"] = [tp[2].sigma for tp in lc["d643"]]
dict["djokovic_time"] = [tp[1] for tp in lc["d643"]]
dict["sampras_mu"] = [tp[2].mu for tp in lc["s402"]]
dict["sampras_sigma"] = [tp[2].sigma for tp in lc["s402"]]
dict["sampras_time"] = [tp[1] for tp in lc["s402"]]
dict["aggasi_mu"] = [tp[2].mu for tp in lc["a092"]]
dict["aggasi_sigma"] = [tp[2].sigma for tp in lc["a092"]]
dict["aggasi_time"] = [tp[1] for tp in lc["a092"]]
dict["vilas_mu"] = [tp[2].mu for tp in lc["v028"]]
dict["vilas_sigma"] = [tp[2].sigma for tp in lc["v028"]]
dict["vilas_time"] = [tp[1] for tp in lc["v028"]]
dict["borg_mu"] = [tp[2].mu for tp in lc["b058"]]
dict["borg_sigma"] = [tp[2].sigma for tp in lc["b058"]]
dict["borg_time"] = [tp[1] for tp in lc["b058"]]
dict["mcenroe_mu"] = [tp[2].mu for tp in lc["m047"]]
dict["mcenroe_sigma"] = [tp[2].sigma for tp in lc["m047"]]
dict["mcenroe_time"] = [tp[1] for tp in lc["m047"]]

maxlen = maximum([length(value) for (key, value) in dict])

using DataFrames
df = DataFrame(Dict(key => [value;repeat([missing],maxlen-length(value))] for (key, value) in dict))

CSV.write("output/atp_learning_curves.csv", df; header=true)

#using Pkg
#Pkg.add("TrueSkillThroughTime")
using CSV
using Dates
using DataFrames
using TrueSkillThroughTime
global const ttt = TrueSkillThroughTime
using DataFrames

base = Dates.value(Date("1970-01-01") - Date("1900-01-01"))

data = CSV.read("input/history.csv", DataFrame, stringtype = String)
djokovic = "d643"
federer = "f324"
sampras = "s402"
lendl = "l018"
connors = "c044"
nadal = "n409"
john_mcenroe = "m047"
bjorn_borg = "b058"
aggasi = "a092"
hewitt = "h432"
edberg = "e004"
vilas = "v028"
nastase = "n008"

courier = "c243"
kuerten = "k293"
murray = "mc10"
wilander = "w023"
roddick = "r485"
#data.w1_id[occursin.("roddick",data.w1_name)]
#data.w1_name[occursin.("pi74",data.w1_id)]




days = Dates.value.(data[:,"time_start"] .- Date("1900-01-01"))
#dates = Date.(Dates.UTD.(days .+ Dates.value(Date("1900-01-01"))))

composition = [ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]   
function fit()
    h = ttt.History(composition=composition, times = days, sigma = 1.6, gamma = 0.036)
    ttt.convergence(h,epsilon=0.01, iterations=10)
    return h
end
h = ttt.History(composition=composition, times = days, sigma = 1.6, gamma = 0.036)
lc = ttt.learning_curves(h)

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

dict["lendl_mu"] = [tp[2].mu for tp in lc["l018"]]
dict["lendl_sigma"] = [tp[2].sigma for tp in lc["l018"]]
dict["lendl_time"] = [tp[1] for tp in lc["l018"]]

dict["connors_mu"] = [tp[2].mu for tp in lc["c044"]]
dict["connors_sigma"] = [tp[2].sigma for tp in lc["c044"]]
dict["connors_time"] = [tp[1] for tp in lc["c044"]]

dict["hewitt_mu"] = [tp[2].mu for tp in lc["h432"]]
dict["hewitt_sigma"] = [tp[2].sigma for tp in lc["h432"]]
dict["hewitt_time"] = [tp[1] for tp in lc["h432"]]

dict["edberg_mu"] = [tp[2].mu for tp in lc["e004"]]
dict["edberg_sigma"] = [tp[2].sigma for tp in lc["e004"]]
dict["edberg_time"] = [tp[1] for tp in lc["e004"]]

dict["nastase_mu"] = [tp[2].mu for tp in lc["n008"]]
dict["nastase_sigma"] = [tp[2].sigma for tp in lc["n008"]]
dict["nastase_time"] = [tp[1] for tp in lc["n008"]]

dict["courier_mu"] = [tp[2].mu for tp in lc["c243"]]
dict["courier_sigma"] = [tp[2].sigma for tp in lc["c243"]]
dict["courier_time"] = [tp[1] for tp in lc["c243"]]

dict["kuerten_mu"] = [tp[2].mu for tp in lc["k293"]]
dict["kuerten_sigma"] = [tp[2].sigma for tp in lc["k293"]]
dict["kuerten_time"] = [tp[1] for tp in lc["k293"]]

dict["murray_mu"] = [tp[2].mu for tp in lc["mc10"]]
dict["murray_sigma"] = [tp[2].sigma for tp in lc["mc10"]]
dict["murray_time"] = [tp[1] for tp in lc["mc10"]]

dict["wilander_mu"] = [tp[2].mu for tp in lc["w023"]]
dict["wilander_sigma"] = [tp[2].sigma for tp in lc["w023"]]
dict["wilander_time"] = [tp[1] for tp in lc["w023"]]

dict["roddick_mu"] = [tp[2].mu for tp in lc["r485"]]
dict["roddick_sigma"] = [tp[2].sigma for tp in lc["r485"]]
dict["roddick_time"] = [tp[1] for tp in lc["r485"]]

maxlen = maximum([length(value) for (key, value) in dict])

df = DataFrame(Dict(key => [value;repeat([missing],maxlen-length(value))] for (key, value) in dict))

CSV.write("output/atp_learning_curves_trueskill.csv", df; header=true)



ttt.convergence(h,epsilon=0.01, iterations=10)    
#@time h = fit()
lc = ttt.learning_curves(h)

top = Set([key  for (key, value) in lc for s in value if ((s[2].mu > 6.0) & (s[1]>base))])



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

dict["lendl_mu"] = [tp[2].mu for tp in lc["l018"]]
dict["lendl_sigma"] = [tp[2].sigma for tp in lc["l018"]]
dict["lendl_time"] = [tp[1] for tp in lc["l018"]]

dict["connors_mu"] = [tp[2].mu for tp in lc["c044"]]
dict["connors_sigma"] = [tp[2].sigma for tp in lc["c044"]]
dict["connors_time"] = [tp[1] for tp in lc["c044"]]

dict["hewitt_mu"] = [tp[2].mu for tp in lc["h432"]]
dict["hewitt_sigma"] = [tp[2].sigma for tp in lc["h432"]]
dict["hewitt_time"] = [tp[1] for tp in lc["h432"]]

dict["edberg_mu"] = [tp[2].mu for tp in lc["e004"]]
dict["edberg_sigma"] = [tp[2].sigma for tp in lc["e004"]]
dict["edberg_time"] = [tp[1] for tp in lc["e004"]]

dict["nastase_mu"] = [tp[2].mu for tp in lc["n008"]]
dict["nastase_sigma"] = [tp[2].sigma for tp in lc["n008"]]
dict["nastase_time"] = [tp[1] for tp in lc["n008"]]

dict["courier_mu"] = [tp[2].mu for tp in lc["c243"]]
dict["courier_sigma"] = [tp[2].sigma for tp in lc["c243"]]
dict["courier_time"] = [tp[1] for tp in lc["c243"]]

dict["kuerten_mu"] = [tp[2].mu for tp in lc["k293"]]
dict["kuerten_sigma"] = [tp[2].sigma for tp in lc["k293"]]
dict["kuerten_time"] = [tp[1] for tp in lc["k293"]]

dict["murray_mu"] = [tp[2].mu for tp in lc["mc10"]]
dict["murray_sigma"] = [tp[2].sigma for tp in lc["mc10"]]
dict["murray_time"] = [tp[1] for tp in lc["mc10"]]

dict["wilander_mu"] = [tp[2].mu for tp in lc["w023"]]
dict["wilander_sigma"] = [tp[2].sigma for tp in lc["w023"]]
dict["wilander_time"] = [tp[1] for tp in lc["w023"]]

dict["roddick_mu"] = [tp[2].mu for tp in lc["r485"]]
dict["roddick_sigma"] = [tp[2].sigma for tp in lc["r485"]]
dict["roddick_time"] = [tp[1] for tp in lc["r485"]]

maxlen = maximum([length(value) for (key, value) in dict])

using DataFrames
df = DataFrame(Dict(key => [value;repeat([missing],maxlen-length(value))] for (key, value) in dict))

CSV.write("output/atp_learning_curves.csv", df; header=true)

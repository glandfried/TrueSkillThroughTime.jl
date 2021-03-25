using CSV
using DataFrames
using Dates
include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill

data = CSV.read("input/history.csv", DataFrame)
times = Dates.value.(data[:,"time_start"] .- Date("1900-1-1"))
composition = [ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]   

sigmas = [i for i in 0.3:0.1:1.2]
gammas = [i for i in 0.0025:0.0025:0.025]

df = DataFrame()
df[!,:rownames] = gammas
for s in sigmas
    df[!,string(s)] .= 0.0
end

for s in 1:length(sigmas)
    for g in 1:length(gammas)
        h = ttt.History(composition=composition, times = times, sigma = sigmas[s], gamma = gammas[g])
        ttt.convergence(h,epsilon=0.01, iterations=10)
        df[g,string(sigmas[s])] = ttt.log_evidence(h)
    end
end

CSV.write("output/atp_optimization.csv", df; header=true)

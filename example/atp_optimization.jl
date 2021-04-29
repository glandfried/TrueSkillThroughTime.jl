using CSV
using DataFrames
using Dates
include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill

data = CSV.read("input/history.csv", DataFrame)
times = Dates.value.(data[:,"time_start"] .- Date("1900-1-1"))
composition = [ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]   

sigmas = [i for i in 1.0:0.05:2.0]
gammas = [i for i in 0.033:0.0005:0.042]

df = DataFrame()
df[!,:rownames] = gammas
for s in sigmas
    df[!,string(s)] .= 0.0
end

for s in 1:length(sigmas)
    for g in 1:length(gammas)
        h = ttt.History(composition=composition, times = times, sigma = sigmas[s], gamma = gammas[g])
        ttt.convergence(h,epsilon=0.01, iterations=5)
        df[g,string(sigmas[s])] = ttt.log_evidence(h)
    end
end

CSV.write("output/atp_optimization.csv", df; header=true)


composition = [ r.double == "t" ? [[r.w1_id,r.w1_id*r.ground,r.w2_id,r.w2_id*r.ground],[r.l1_id,r.l1_id*r.ground,r.l2_id,r.l2_id*r.ground]] : [[r.w1_id,r.w1_id*r.ground],[r.l1_id,r.l1_id*r.ground]] for r in eachrow(data) ]   

players = Set(vcat(([ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]...)...))

sigmas = [i for i in 0.25:0.05:0.45]
gammas = [i for i in 0.000:0.001:0.002]

df = DataFrame()
df[!,:rownames] = gammas
for s in sigmas
    df[!,string(s)] .= 0.0
end

#s=0.35;g=0.0; sp=1.6; gp=0.036
function scenario(s,g, sp, gp)
    priors = Dict([(p, ttt.Player(ttt.Gaussian(0., sp), 1.0, gp) ) for p in players])
    h = ttt.History(composition=composition, times = times, sigma = s, gamma = g, beta = 0.0, priors = priors)
    ttt.convergence(h,epsilon=0.01, iterations=2)
    return ttt.log_evidence(h,online=false,agents=["n409"])
end

#scenario(0.35, 0.000, 1.6, 0.036)
for s in 1:length(sigmas)
    for g in 1:length(gammas)
        GC.gc()
        df[g,string(sigmas[s])] = scenario(sigmas[s],gammas[g], 1.6, 0.036)
        println("s:",sigmas[s], ", g:", gammas[g], ". ", df[g,string(sigmas[s])])
    end
end


CSV.write("output/atp_ground_optimization.csv", df; header=true)


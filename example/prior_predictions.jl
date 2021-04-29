using DataFrames
using CSV
include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test


p_d_m = [log(0.5)]
p_d_m_trueskill = [log(0.5)]
p_d_m_hat = [log(0.5)]
loocv_hat = [log(0.5)]
composition = Vector{Vector{Vector{String}}}()            
push!(composition, [["b"],["a"]])
h = ttt.History(composition)
for i in 1:50
    lc = ttt.learning_curves(h)    
    push!(p_d_m, p_d_m[end]+log(ttt.Game([[ttt.Player(lc["a"][end][2])],[ttt.Player(lc["b"][end][2])]]).evidence))
    push!(composition, [["a"],["b"]])
    h = ttt.History(composition)
    push!(p_d_m_trueskill,ttt.log_evidence(h))
    ttt.convergence(h,iterations=10)
    push!(loocv_hat,ttt.log_evidence(h))
    push!(p_d_m_hat,ttt.log_evidence(h, online=true))
    
    lc = ttt.learning_curves(h)    
    push!(p_d_m, p_d_m[end]+log(ttt.Game([[ttt.Player(lc["b"][end][2])],[ttt.Player(lc["b"][end][2])]]).evidence))
    push!(composition, [["b"],["a"]])
    h = ttt.History(composition)
    push!(p_d_m_trueskill,ttt.log_evidence(h))
    ttt.convergence(h,iterations=10)
    push!(loocv_hat,ttt.log_evidence(h))
    push!(p_d_m_hat,ttt.log_evidence(h, online=true))
end

@test sum(p_d_m_trueskill .-1e-4 .<= p_d_m) == length(p_d_m)

@test sum(p_d_m_hat .-1e-4 .<= p_d_m) == length(p_d_m)

#@test sum(loocv_hat .-1e-4  .<= p_d_m ) == length(p_d_m)

df = DataFrame()
df[!,:p_d_m] = p_d_m
df[!,:p_d_m_trueskill] = p_d_m_trueskill
df[!,:p_d_m_hat] = p_d_m_hat
df[!,:loocv_hat] = loocv_hat

CSV.write("output/prior_predictions.csv", df; header=true)

# using Plots
# plot(loocv_hat-p_d_m)
# plot!(p_d_m_hat-p_d_m)
# plot!(p_d_m_trueskill-p_d_m)

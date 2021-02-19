using CSV
using DataFrames
using Dates
include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill

data = CSV.read("input/history.csv")
times = Dates.value.(data[:,"time_start"] .- Date("1900-1-1"))
composition = [ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] : [[r.w1_id],[r.l1_id]] for r in eachrow(data) ]   
h = ttt.History(composition=composition, times = times, iterations = 16)
ttt.convergence(h,true)

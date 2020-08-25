include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test
using CSV
using JLD2

@testset "Test OGS" begin
    data = CSV.read("summary_filtered.csv")
    prior_dict = Dict{String,ttt.Rating}()
    for h_key in Set([(row.handicap, row.width) for row in eachrow(data) ])
        prior_dict[string(h_key)] = ttt.Rating(0.,25.0/3.,0.,1.0/100)
    end
    results = [row.black_win == 1 ? [1,0] : [0, 1] for row in eachrow(data) ]
    composition = [ r.handicap<2 ? [[string(r.white)],[string(r.black)]] : [[string(r.white)],[string(r.black),string((r.handicap,r.width))]] for r in eachrow(data) ]   
    times = Vector{Int64}()
    
    println(now())
    
    h = ttt.History(composition, results, times , prior_dict)
    
    println(now())
    
    @save "ogs_history.jld2" h
    
    w_mean = [ ttt.posterior(h.batches[r.row], string(r.white)).mu  for r in eachrow(data) ]                                                            
    b_mean = [ ttt.posterior(h.batches[r.row], string(r.black)).mu  for r in eachrow(data) ]                                                            
    w_std = [ ttt.posterior(h.batches[r.row], string(r.white)).sigma  for r in eachrow(data) ]                                                            
    b_std = [ ttt.posterior(h.batches[r.row], string(r.black)).sigma  for r in eachrow(data) ]                                                            
    
    h_mean = [ r.handicap > 1 ? ttt.posterior(h.batches[r.row] ,string((r.hhandicap,r.width))).mu : 0 for r in eachrow(data) ]
    h_std = [ r.handicap > 1 ? ttt.posterior(h.batches[r.row] ,string((r.hhandicap,r.width))).sigma : 0 for r in eachrow(data) ]
    evidence = [ h.batches[r.row].evidences[1] for r in eachrow(data) ] 
    
    @save "ogs_estimations.jld2" w_mean b_mean w_std b_std h_mean h_std evidence
    
end


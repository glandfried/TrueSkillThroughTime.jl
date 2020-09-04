include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test
using CSV
using JLD2
using Dates
using DataFrames

@testset "Test OGS" begin
    data = CSV.read("data/summary_filtered.csv")    
    prior_dict = Dict{String,ttt.Rating}()
    for h_key in Set([(row.handicap, row.width) for row in eachrow(data) ])
        prior_dict[string(h_key)] = ttt.Rating(0.,25.0/3.,0.,1.0/100)
    end
    results = [row.black_win == 1 ? [1,0] : [0, 1] for row in eachrow(data) ]
    events = [ r.handicap<2 ? [[string(r.white)],[string(r.black)]] : [[string(r.white)],[string(r.black),string((r.handicap,r.width))]] for r in eachrow(data) ]   
    #times = [0 for _ in 1:length(events)] # To test batch performance
    times = Vector{Int64}()
    
    println(now())
    
    h = ttt.History(events, results, times , prior_dict)
    
    ts_log_evidence = ttt.log_evidence(h)
    
    println(now())
    
    ttt.convergence(h)
    
    ttt_log_evidence = ttt.log_evidence(h)
    
    println(now())
    
    print("TS: ", ts_log_evidence, ", TTT:", ttt_log_evidence)
    
    @test ts_log_evidence < ttt_log_evidence
    
    w_mean = [ ttt.posterior(h.batches[r], string(data[r,"white"])).mu for r in 1:size(data)[1]]                                                            
    b_mean = [ ttt.posterior(h.batches[r], string(data[r,"black"])).mu  for r in 1:size(data)[1]]                                                            
    w_std = [ ttt.posterior(h.batches[r], string(data[r,"white"])).sigma for r in 1:size(data)[1]]                                                            
    b_std = [ ttt.posterior(h.batches[r], string(data[r,"black"])).sigma for r in 1:size(data)[1]]                                                            
    
    h_mean = [ data[r,"handicap"] > 1 ? ttt.posterior(h.batches[r] ,string((data[r,"handicap"],data[r,"width"]))).mu : 0.0 for r in 1:size(data)[1]]
    h_std = [ data[r,"handicap"] > 1 ? ttt.posterior(h.batches[r] ,string((data[r,"handicap"],data[r,"width"]))).sigma : 0.0 for r in 1:size(data)[1]]
    evidence = [ h.batches[r].evidences[1] for r in 1:size(data)[1]] 
    
    df = DataFrame(id = data[:"id"]
                  ,white = data[:"white"]
                  ,black = data[:"black"]
                  ,handicap = data[:"handicap"]
                  ,width = data[:"width"]
                  ,w_mean = w_mean
                  ,b_mean = b_mean
                  ,w_std = w_std
                  ,b_std = b_std
                  ,h_mean = h_mean
                  ,h_std = h_std
                  ,evidence = evidence)
    
    CSV.write("data/longtest_output.csv", df; header=true)
    
end

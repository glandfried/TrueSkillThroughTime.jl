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
    prior_copy = copy(prior_dict)
    results = [row.black_win == 1 ? [1,0] : [0, 1] for row in eachrow(data) ]
    events = [ r.handicap<2 ? [[string(r.white)],[string(r.black)]] : [[string(r.white)],[string(r.black),string((r.handicap,r.width))]] for r in eachrow(data) ]   
    times = Vector{Int64}()
    
    
    println(now())
    h = ttt.History(events, results, times , prior_dict, ttt.Environment(mu=0.0,sigma=10.,beta=1.,gamma=0.2,iter=16))
    println(now())
    ttt.convergence(h)
    gammas = [(0.00, -Inf), (0.2,ttt.log_evidence(h)), (0.40, -Inf)]        
    println(now())
    
    delta = Inf::Float64
    while delta > 0.001
        
        gamma_left = (gammas[1][1]+gammas[2][1])/2
        
        delta = gammas[2][1] - gamma_left 
        
        prior_dict = copy(prior_copy)
        println("Gamma left = ", gamma_left)
        println(now())
        h = ttt.History(events, results, times , prior_dict, ttt.Environment(mu=0.,sigma=10.,beta=1.,gamma=gamma_left,iter=16))
        ts_log_evidence_right = ttt.log_evidence(h)
        println(now())
        ttt.convergence(h)
        log_evidence_left = ttt.log_evidence(h)
        println(now())
        
        gamma_right = (gammas[2][1]+gammas[3][1])/2
        
        prior_dict = copy(prior_copy)
        println("Gamma right = ", gamma_right)
        println(now())
        h = ttt.History(events, results, times , prior_dict, ttt.Environment(mu=0.,sigma=10.,beta=1.,gamma=gamma_right,iter=16))
        ts_log_evidence_right = ttt.log_evidence(h)
        println(now())
        ttt.convergence(h)
        log_evidence_right = ttt.log_evidence(h)
        println(now())
        
        wm = argmax([log_evidence_left ,  gammas[2][2] ,log_evidence_right ])
        
        if wm == 1
            gammas = [gammas[1], (gamma_left, log_evidence_left) ,gammas[2]]
        elseif wm == 2
            gammas = [(gamma_left, log_evidence_left) ,gammas[2], (gamma_right, log_evidence_right)]
        elseif wm == 3
            gammas = [gammas[2], (gamma_right, log_evidence_right), gammas[3]]
        end
        
    end
    
    @test isapprox(gammas[2][1], 0.2125)
    ttt_log_evidence = gammas[2][2]
    #print("TS: ", ts_log_evidence, ", TTT:", ttt_log_evidence)
    #@test ts_log_evidence < ttt_log_evidence
    
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
    CSV.write("data/longtest_gamma.csv", DataFrame(gamma = gammas[2][1], log_evidence = gammas[2][2]); header=true)
    
end


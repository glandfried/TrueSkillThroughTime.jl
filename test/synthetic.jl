include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test
using CSV
using DataFrames

    
@testset "All" begin
    @testset "synthetic/same_strength.csv" begin 
        events = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ,[["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]]]
        results = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]    
        priors = Dict{String,ttt.Rating}()
        for k in ["aj", "bj", "cj"]
            priors[k] = ttt.Rating(0., 3., 0.5, 0.0, k ) 
        end
        h = ttt.History(events, results, [1,2,3,4,5,6], priors)
        lc_ts = ttt.learning_curves(h)
        ttt.convergence(h)
        lc_ttt = ttt.learning_curves(h)
        
        df = DataFrame(mu_a_ts = [ N.mu for (k,N) in lc_ts["aj"]]
                      ,sigma_a_ts = [ N.sigma for (k,N) in lc_ts["aj"]]
                      ,mu_a_ttt = [ N.mu for (k,N) in lc_ttt["aj"]]
                      ,sigma_a_ttt = [ N.sigma for (k,N) in  lc_ttt["aj"]]
                      ,time_a = [ k for (k,N) in lc_ts["aj"]]
                      ,mu_b_ts = [ N.mu for (k,N) in lc_ts["bj"]]
                      ,sigma_b_ts = [ N.sigma for (k,N) in lc_ts["bj"]]
                      ,mu_b_ttt = [ N.mu for (k,N) in lc_ttt["bj"]]
                      ,sigma_b_ttt = [ N.sigma for (k,N) in lc_ttt["bj"]]
                      ,time_b = [ k for (k,N) in lc_ts["bj"]]
                      ,mu_c_ts = [ N.mu for (k,N) in lc_ts["cj"]]
                      ,sigma_c_ts = [ N.sigma for (k,N) in lc_ts["cj"]]
                      ,mu_c_ttt = [ N.mu for (k,N) in lc_ttt["cj"]]
                      ,sigma_c_ttt = [ N.sigma for (k,N) in lc_ttt["cj"]]
                      ,time_c = [ k for (k,N) in lc_ts["cj"]]
                      )
    
        CSV.write("synthetic/same_strength.csv", df; header=true)
        @test true
    end
    @testset "synthetic/same_strength_two_groups.csv" begin 
        predicciones_mle = Float64[] 
        predicciones_aiaj = Float64[] 
        predicciones_bibj = Float64[] 
        predicciones_aibi = Float64[] 
        predicciones_bici = Float64[] 
        predicciones_ajbj = Float64[] 
        predicciones_bjcj = Float64[] 
        events = [ [["aj"],["bj"]], [["bj"],["cj"]], [["cj"],["aj"]],[["aj"],["bj"]], [["bj"],["cj"]], [["cj"],["aj"]],
                   [["ai"],["bi"]], [["bi"],["ci"]], [["ci"],["ai"]], [["ai"],["bi"]], [["bi"],["ci"]], [["ci"],["ai"]],
                   [["ai"],["aj"]] ]
        results = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1], [0,1],[0,1],[0,1],[0,1],[0,1],[0,1], [1,0]]    
        batches = [1,1,1,1,1,1,1,1,1,1,1,1,1]
        Nbeta = ttt.Gaussian(0.0,0.5)
            
        for i in 1:20
            push!(events, [["ai"],["aj"]])
            push!(results, [0,1])
            push!(batches, 1)
            priors = Dict{String,ttt.Rating}()
            for k in ["aj", "bj", "cj", "ai", "bi", "ci"]
                priors[k] = ttt.Rating(0., 5.0, 0.5, 0.0, k ) 
            end
            
            h = ttt.History(events, results, batches, priors)
            ttt.convergence(h)
            ttt.convergence(h)
            ttt.convergence(h)
            ttt.convergence(h)
            
            lc = ttt.learning_curves(h)
            
            push!(predicciones_mle, i/(1.0+i))
            
            push!(predicciones_aiaj, 1-ttt.cdf(lc["ai"][1][2]+Nbeta - lc["aj"][1][2]+Nbeta, 0.))
            push!(predicciones_bibj, 1-ttt.cdf(lc["bi"][1][2]+Nbeta - lc["bj"][1][2]+Nbeta, 0.))
            push!(predicciones_aibi, 1-ttt.cdf(lc["ai"][1][2]+Nbeta - lc["bi"][1][2]+Nbeta, 0.))
            push!(predicciones_bici, 1-ttt.cdf(lc["bi"][1][2]+Nbeta - lc["ci"][1][2]+Nbeta, 0.))
            push!(predicciones_ajbj, 1-ttt.cdf(lc["aj"][1][2]+Nbeta - lc["bj"][1][2]+Nbeta, 0.))
            push!(predicciones_bjcj, 1-ttt.cdf(lc["bj"][1][2]+Nbeta - lc["cj"][1][2]+Nbeta, 0.))
        end
        df = DataFrame(mle = predicciones_mle
                      ,aiaj = predicciones_aiaj
                      ,bibj = predicciones_bibj
                      ,aibi = predicciones_aibi
                      ,bici = predicciones_bici
                      )
        
        CSV.write("synthetic/same_strength_two_groups.csv", df; header=true)
        @test true
    end
    @testset "synthetic/smoothing.csv" begin
        events = [ [["a"],["b"]], [["a"],["b"]]]
        results = [[0,1],[1,0]]
        times = [1,2]
        priors = Dict{String,ttt.Rating}()
        Nbeta = ttt.Gaussian(0.,0.5)
        for k in ["a", "b"]
            priors[k] = ttt.Rating(0., 3., 0.5, 0.0, k ) 
        end
        h = ttt.History(events, results, times, priors)
        fp_a_1 = Vector{ttt.Gaussian}()
        bp_a_1 = Vector{ttt.Gaussian}() 
        lh_a_1 = Vector{ttt.Gaussian}()
        wp_a_1 = Vector{ttt.Gaussian}()

        fp_a_2 = Vector{ttt.Gaussian}()
        bp_a_2 = Vector{ttt.Gaussian}() 
        lh_a_2 = Vector{ttt.Gaussian}()
        wp_a_2 = Vector{ttt.Gaussian}()
        
        lh_b_1 = Vector{ttt.Gaussian}()
        lh_b_2 = Vector{ttt.Gaussian}()
        
        wp_b_1 = Vector{ttt.Gaussian}()
        wp_b_2 = Vector{ttt.Gaussian}()
        
        p_a_1 =  Vector{ttt.Gaussian}()
        p_a_2 =  Vector{ttt.Gaussian}()
        p_b_1 =  Vector{ttt.Gaussian}()
        p_b_2 =  Vector{ttt.Gaussian}()
        
        
        e_1 = Float64[]
        e_2 = Float64[]
        d_div_1 = Vector{ttt.Gaussian}()
        d_div_2 = Vector{ttt.Gaussian}()
        
        push!(fp_a_1, h.batches[1].prior_forward["a"].N)
        push!(bp_a_1, h.batches[1].prior_backward["a"])
        push!(lh_a_1, h.batches[1].likelihoods[1][1][1])
        push!(wp_a_1, fp_a_1[end]*bp_a_1[end])
        push!(wp_b_1, h.batches[1].prior_forward["b"].N*h.batches[1].prior_backward["b"])
        push!(fp_a_2, h.batches[2].prior_forward["a"].N)
        push!(bp_a_2, h.batches[2].prior_backward["a"])
        push!(lh_a_2, h.batches[2].likelihoods[1][1][1])
        push!(wp_a_2, fp_a_2[end]*bp_a_2[end])
        push!(wp_b_2, h.batches[2].prior_forward["b"].N*h.batches[2].prior_backward["b"])
        push!(e_1, h.batches[1].evidences[1])
        push!(e_2, h.batches[2].evidences[1])
        push!(lh_b_1, h.batches[1].likelihoods[1][2][1])
        push!(lh_b_2, h.batches[2].likelihoods[1][2][1])
        
        push!(p_a_1, wp_a_1[end]*lh_a_1[end])
        push!(p_a_2, wp_a_2[end]*lh_a_2[end])
        push!(p_b_1, wp_b_1[end]*lh_b_1[end])
        push!(p_b_2, wp_b_2[end]*lh_b_2[end])
        
        
        d_1 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
        push!(d_div_1 , ttt.trunc(d_1,0.,false)/d_1)
        d_2 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
        push!(d_div_2 , ttt.trunc(d_2,0.,false)/d_2)
        
        for _ in 1:10
            ttt.iteration(h)
            push!(fp_a_1, h.batches[1].prior_forward["a"].N)
            push!(bp_a_1, h.batches[1].prior_backward["a"])
            push!(lh_a_1, h.batches[1].likelihoods[1][1][1])
            push!(wp_a_1, fp_a_1[end]*bp_a_1[end])
            push!(wp_b_1, h.batches[1].prior_forward["b"].N*h.batches[1].prior_backward["b"])
            push!(fp_a_2, h.batches[2].prior_forward["a"].N)
            push!(bp_a_2, h.batches[2].prior_backward["a"])
            push!(lh_a_2, h.batches[2].likelihoods[1][1][1])
            push!(wp_a_2, fp_a_2[end]*bp_a_2[end])
            push!(wp_b_2, h.batches[2].prior_forward["b"].N*h.batches[2].prior_backward["b"])
            push!(e_1, h.batches[1].evidences[1])
            push!(e_2, h.batches[2].evidences[1])
            push!(lh_b_1, h.batches[1].likelihoods[1][2][1])
            push!(lh_b_2, h.batches[2].likelihoods[1][2][1])
            d_1 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
            push!(d_div_1 , ttt.trunc(d_1,0.,false)/d_1)
            d_2 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
            push!(d_div_2 , ttt.trunc(d_2,0.,false)/d_2)
            push!(p_a_1, wp_a_1[end]*lh_a_1[end])
            push!(p_a_2, wp_a_2[end]*lh_a_2[end])
            push!(p_b_1, wp_b_1[end]*lh_b_1[end])
            push!(p_b_2, wp_b_2[end]*lh_b_2[end])
        end
        
        df = DataFrame(fp_a_1_mu = [N.mu for N in fp_a_1]
                      ,fp_a_1_sigma = [N.sigma for N in fp_a_1]
                      ,bp_a_1_mu = [N.mu for N in bp_a_1]
                      ,bp_a_1_sigma = [N.sigma for N in bp_a_1]
                      ,lh_a_1_mu = [N.mu for N in lh_a_1]
                      ,lh_a_1_sigma = [N.sigma for N in lh_a_1]
                      ,wp_a_1_mu = [N.mu for N in wp_a_1]
                      ,wp_a_1_sigma = [N.sigma for N in wp_a_1]
                      ,p_a_1_mu = [N.mu for N in p_a_1]
                      ,p_a_1_sigma = [N.sigma for N in p_a_1]
                      ,p_a_2_mu = [N.mu for N in p_a_2]
                      ,p_a_2_sigma = [N.sigma for N in p_a_2]
                      ,p_b_1_mu = [N.mu for N in p_b_1]
                      ,p_b_1_sigma = [N.sigma for N in p_b_1]
                      ,p_b_2_mu = [N.mu for N in p_b_2]
                      ,p_b_2_sigma = [N.sigma for N in p_b_2]
                      ,wp_b_1_mu = [N.mu for N in wp_b_1]
                      ,wp_b_1_sigma = [N.sigma for N in wp_b_1]
                      ,fp_a_2_mu = [N.mu for N in fp_a_2]
                      ,fp_a_2_sigma = [N.sigma for N in fp_a_2]
                      ,bp_a_2_mu = [N.mu for N in bp_a_2]
                      ,bp_a_2_sigma = [N.sigma for N in bp_a_2]
                      ,lh_a_2_mu = [N.mu for N in lh_a_2]
                      ,lh_a_2_sigma = [N.sigma for N in lh_a_2]
                      ,wp_a_2_mu = [N.mu for N in wp_a_2]
                      ,wp_a_2_sigma = [N.sigma for N in wp_a_2]
                      ,wp_b_2_mu = [N.mu for N in wp_b_2]
                      ,wp_b_2_sigma = [N.sigma for N in wp_b_2]
                      ,lh_b_1_mu = [N.mu for N in lh_b_1]
                      ,lh_b_2_mu = [N.mu for N in lh_b_2]
                      ,lh_b_1_sigma = [N.sigma for N in lh_b_1]
                      ,lh_b_2_sigma = [N.sigma for N in lh_b_2]
                      ,e_1 = e_1
                      ,e_2 = e_2 
                      ,d_div_1_mu = [N.mu for N in d_div_1]
                      ,d_div_1_sigma = [N.sigma for N in d_div_1]
                      ,d_div_2_mu = [N.mu for N in d_div_2]
                      ,d_div_2_sigma = [N.sigma for N in d_div_2]
                      )
        CSV.write("synthetic/smoothing.csv", df; header=true)
        @test true
    end
    @testset "synthetic/ttt_vs_ts.csv" begin
        
        events = Array{Array{Array{String,1},1},1}()
        results =  Array{Array{Int64,1},1}()
        times = Int64[]
        ts_log_evidence = Float64[]
        ttt_log_evidence = Float64[]
        
        ts_last = Float64[]; ts_midle = Float64[]; ts_second = Float64[]
        ttt_last = Float64[]; ttt_midle = Float64[]; ttt_second = Float64[]
        for i in 0:127
            push!(events ,  [["a"],["b"]], [["a"],["b"]])
            push!(results ,  [0,1],[1,0] )
            push!(times,  i*2+1, i*2+2 )
            priors = Dict{String,ttt.Rating}()
            for k in ["a", "b"]
                priors[k] = ttt.Rating(0., 3.0, 0.5, 0.0, k ) 
            end
            h = ttt.History(events, results, times, priors)
            push!(ts_log_evidence , ttt.log_evidence(h))
            push!(ts_last, h.batches[end].evidences[1]); push!(ts_midle, h.batches[i*2+1].evidences[1]); 
            push!(ts_second, h.batches[2].evidences[1]);
            ttt.convergence(h)
            push!(ttt_last, h.batches[end].evidences[1]); push!(ttt_midle, h.batches[i*2+1].evidences[1]); 
            push!(ttt_log_evidence, ttt.log_evidence(h))
            push!(ttt_second, h.batches[2].evidences[1])
        end
       df = DataFrame(ts_log_evidence = ts_log_evidence
                     ,ttt_log_evidence = ttt_log_evidence
                     ,ts_last = ts_last
                     ,ts_midle = ts_midle
                     ,ts_second = ts_second
                     ,ttt_second = ttt_second
                     ,ttt_last = ttt_last
                     ,ttt_midle = ttt_midle )
            
        CSV.write("synthetic/ttt_vs_ts.csv", df; header=true)
        @test true
    end
    @testset "Best gamma" begin
        
        function skill(exp::Int64, alpha::Float64=0.133)
            return exp^alpha - 1
        end
        mean_agent = [skill(i) for i in 1:1000] 
        beta = 0.5
        
        using Random
        Random.seed!(1)
        
        mean_target = [(Random.randn(1)[1]*beta + skill(i)) for i in 1:1000] 
        perf_target = [(Random.randn(1)[1]*beta + mean_target[i]) for i in 1:1000] 
        perf_agent = [(Random.randn(1)[1]*beta + mean_target[i]) for i in 1:1000] 
        events = [ [["a"], [string(i)] ] for i in 1:1000]
        results = [ perf_agent[i] > perf_target[i] ? [0,1] : [1,0] for i in 1:1000 ]
        batches= [i for i in 1:1000 ]
        
    
        selected_gammas = [0.005,0.01,0.015,0.02,0.025]
        
        gammas = [gamma for gamma in 0.001:0.001:0.04]
        evidencias_ts = Float64[]
        evidencias_ttt = Float64[]
        learning_curves = Vector{Vector{Float64}}()
        for gamma in gammas#gamma=0.015
            
            priors = Dict{String,ttt.Rating}()
            priors["a"] = ttt.Rating(0., 3.0, beta, gamma, "a") 
            for k in 1:1000
                priors[string(k)] = ttt.Rating(mean_target[k], 0.5, beta, 0.0, string(k) ) 
            end
            h = ttt.History(events, results, batches, priors)
            push!(evidencias_ts, ttt.log_evidence(h))
            ttt.convergence(h)
            push!(evidencias_ttt, ttt.log_evidence(h))
            if gamma in selected_gammas
                push!(learning_curves, [r.mu for (t, r) in ttt.learning_curves(h)["a"]] )
            end
        end
        #using Plots
        #plot([g for g in gammas ],evidencias_ts)
        #plot!([g for g in gammas ] ,evidencias_ttt)
        
        @test 0.015 == gammas[argmax(evidencias_ttt)]
    
        
        df = DataFrame(gammas = gammas
                      ,evidencias_ts = evidencias_ts
                      ,evidencias_ttt = evidencias_ttt)
        CSV.write("synthetic/best_gamma-evidences.csv", df; header=true)
       
        df = DataFrame(mean_agent = mean_agent 
                      ,lc_005 = learning_curves[1]
                      ,lc_01  = learning_curves[2]
                      ,lc_015 = learning_curves[3]
                      ,lc_020 = learning_curves[4]
                      ,lc_025 = learning_curves[5])
        CSV.write("synthetic/best_gamma-learning_curves.csv", df; header=true)
       
        #plot(mean_agent)
        #plot!(learning_curves[3])
    end
            
    
end

include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test
using CSV
using DataFrames

@testset "Examples" begin

@testset "same_strength" begin
    events = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ,[["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]]]
    results = [[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.]]    
    priors = Dict{String,ttt.Player}()
    for k in ["aj", "bj", "cj"]
        priors[k] = ttt.Player(ttt.Gaussian(0., 3.), 0.5, 0.0) 
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

    CSV.write("output/same_strength.csv", df; header=true)
    @test true
end
@testset "same_strength_two_groups" begin
    predicciones_mle = Float64[] 
    predicciones_aiaj = Float64[] 
    predicciones_bibj = Float64[] 
    predicciones_aibi = Float64[] 
    predicciones_bici = Float64[] 
    predicciones_ajbj = Float64[] 
    predicciones_bjcj = Float64[] 
    composition = [ [["aj"],["bj"]], [["bj"],["cj"]], [["cj"],["aj"]],[["aj"],["bj"]], [["bj"],["cj"]], [["cj"],["aj"]],
                [["ai"],["bi"]], [["bi"],["ci"]], [["ci"],["ai"]], [["ai"],["bi"]], [["bi"],["ci"]], [["ci"],["ai"]],
                [["aj"],["ai"]] ]
    times = [1,1,1,1,1,1,1,1,1,1,1,1,1]
    Nbeta = ttt.Gaussian(0.0,0.5)
        
    for i in 1:150
        push!(composition, [["ai"],["aj"]])
        push!(times, 1)
        priors = Dict{String,ttt.Player}()
        for k in ["aj", "bj", "cj", "ai", "bi", "ci"]
            priors[k] = ttt.Player(ttt.Gaussian(0., 5.0), 0.5, 0.0) 
        end
        
        h = ttt.History(composition=composition, times=times, priors=priors, iter=40)
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
    
    # S - Shape
    using Plots
    plot(predicciones_mle,predicciones_mle,legend=(-10.0, 10.0))
    plot!(predicciones_mle,predicciones_aiaj)
    #plot!(predicciones_mle,predicciones_bibj)
    
    df = DataFrame(mle = predicciones_mle
                    ,aiaj = predicciones_aiaj
                    ,bibj = predicciones_bibj
                    ,aibi = predicciones_aibi
                    ,bici = predicciones_bici
                    )
    
    CSV.write("output/same_strength_two_groups.csv", df; header=true)
    @test true
end
@testset "smoothing" begin
    events = [ [["a"],["b"]], [["a"],["b"]]]
    results = [[1.,0.],[0.,1.]]
    times = [1,2]
    priors = Dict{String,ttt.Player}()
    Nbeta = ttt.Gaussian(0.,0.5)
    for k in ["a", "b"]
        priors[k] = ttt.Player(ttt.Gaussian(0., 3.), 0.5, 0.0 ) 
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
    
    push!(fp_a_1, h.batches[1].skills["a"].forward)
    push!(bp_a_1, h.batches[1].skills["a"].backward)
    push!(lh_a_1, h.batches[1].skills["a"].likelihood)
    push!(wp_a_1, fp_a_1[end]*bp_a_1[end])
    push!(wp_b_1, h.batches[1].skills["b"].forward*h.batches[1].skills["b"].backward)
    push!(fp_a_2, h.batches[2].skills["a"].forward)
    push!(bp_a_2, h.batches[2].skills["a"].backward)
    push!(lh_a_2, h.batches[2].skills["a"].likelihood)
    push!(wp_a_2, fp_a_2[end]*bp_a_2[end])
    push!(wp_b_2, h.batches[2].skills["b"].forward*h.batches[2].skills["b"].backward)
    push!(e_1, h.batches[1].events[1].evidence)
    push!(e_2, h.batches[2].events[1].evidence)
    push!(lh_b_1, h.batches[1].skills["b"].likelihood)
    push!(lh_b_2, h.batches[2].skills["b"].likelihood)
    
    push!(p_a_1, wp_a_1[end]*lh_a_1[end])
    push!(p_a_2, wp_a_2[end]*lh_a_2[end])
    push!(p_b_1, wp_b_1[end]*lh_b_1[end])
    push!(p_b_2, wp_b_2[end]*lh_b_2[end])
    
    
    d_1 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
    push!(d_div_1 , ttt.approx(d_1,0.,false)/d_1)
    d_2 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
    push!(d_div_2 , ttt.approx(d_2,0.,false)/d_2)
    
    for _ in 1:10
        ttt.iteration(h)
        push!(fp_a_1, h.batches[1].skills["a"].forward)
        push!(bp_a_1, h.batches[1].skills["a"].backward)
        push!(lh_a_1, h.batches[1].skills["a"].likelihood)
        push!(wp_a_1, fp_a_1[end]*bp_a_1[end])
        push!(wp_b_1, h.batches[1].skills["b"].forward*h.batches[1].skills["b"].backward)
        push!(fp_a_2, h.batches[2].skills["a"].forward)
        push!(bp_a_2, h.batches[2].skills["a"].backward)
        push!(lh_a_2, h.batches[2].skills["a"].likelihood)
        push!(wp_a_2, fp_a_2[end]*bp_a_2[end])
        push!(wp_b_2, h.batches[2].skills["b"].forward*h.batches[2].skills["b"].backward)
        push!(e_1, h.batches[1].events[1].evidence)
        push!(e_2, h.batches[2].events[1].evidence)
        push!(lh_b_1, h.batches[1].skills["b"].likelihood)
        push!(lh_b_2, h.batches[2].skills["b"].likelihood)
        
        d_1 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
        push!(d_div_1 , ttt.approx(d_1,0.,false)/d_1)
        d_2 = wp_a_1[end]+Nbeta - wp_b_1[end]+Nbeta 
        push!(d_div_2 , ttt.approx(d_2,0.,false)/d_2)
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
    CSV.write("output/smoothing.csv", df; header=true)
    @test true
end
@testset "ttt_vs_ts" begin
    
    events = Array{Array{Array{String,1},1},1}()
    results =  Array{Array{Float64,1},1}()
    times = Int64[]
    ts_log_evidence = Float64[]
    ttt_log_evidence = Float64[]
    
    ts_last = Float64[]; ts_midle = Float64[]; ts_second = Float64[]
    ttt_last = Float64[]; ttt_midle = Float64[]; ttt_second = Float64[]
    for i in 0:127
        push!(events ,  [["a"],["b"]], [["a"],["b"]])
        push!(results ,  [1.,0.],[0.,1.] )
        push!(times,  i*2+1, i*2+2 )
        priors = Dict{String,ttt.Player}()
        for k in ["a", "b"]
            priors[k] = ttt.Player(ttt.Gaussian(0., 3.0), 0.5, 0.0) 
        end
        h = ttt.History(events, results, times, priors)
        push!(ts_log_evidence , ttt.log_evidence(h))
        push!(ts_last, h.batches[end].events[1].evidence); push!(ts_midle, h.batches[i*2+1].events[1].evidence); 
        push!(ts_second, h.batches[2].events[1].evidence);
        ttt.convergence(h)
        push!(ttt_last, h.batches[end].events[1].evidence); push!(ttt_midle, h.batches[i*2+1].events[1].evidence); 
        push!(ttt_log_evidence, ttt.log_evidence(h))
        push!(ttt_second, h.batches[2].events[1].evidence)
    end
    df = DataFrame(ts_log_evidence = ts_log_evidence
                    ,ttt_log_evidence = ttt_log_evidence
                    ,ts_last = ts_last
                    ,ts_midle = ts_midle
                    ,ts_second = ts_second
                    ,ttt_second = ttt_second
                    ,ttt_last = ttt_last
                    ,ttt_midle = ttt_midle )
        
    CSV.write("output/ttt_vs_ts.csv", df; header=true)
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
    results = [ perf_agent[i] > perf_target[i] ? [1.,0.] : [0.,1.] for i in 1:1000 ]
    batches= [i for i in 1:1000 ]
    
    selected_gammas = [0.005,0.01,0.015,0.02,0.025]
    
    gammas = [gamma for gamma in 0.001:0.001:0.04]
    evidencias_ts = Float64[]
    evidencias_ttt = Float64[]
    learning_curves = Vector{Vector{Float64}}()
    for gamma in gammas#gamma=0.015
        
        priors = Dict{String,ttt.Player}()
        priors["a"] = ttt.Player(ttt.Gaussian(0., 3.0), beta, gamma) 
        for k in 1:1000
            priors[string(k)] = ttt.Player(ttt.Gaussian(mean_target[k], 0.5), beta, 0.0) 
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
    
    #plot(mean_agent )
    #plot!(learning_curves[3])
    
    
    @test 0.015 == gammas[argmax(evidencias_ttt)]
    
    
    df = DataFrame(gammas = gammas
                    ,evidencias_ts = evidencias_ts
                    ,evidencias_ttt = evidencias_ttt)
    CSV.write("output/best_gamma-evidences.csv", df; header=true)
    
    df = DataFrame(mean_agent = mean_agent 
                    ,lc_005 = learning_curves[1]
                    ,lc_01  = learning_curves[2]
                    ,lc_015 = learning_curves[3]
                    ,lc_020 = learning_curves[4]
                    ,lc_025 = learning_curves[5])
    CSV.write("output/best_gamma-learning_curves.csv", df; header=true)
    
    #plot(mean_agent)
    #plot!(learning_curves[3])
end
@testset "Chain of betas" begin
    using Random
    Random.seed!(1)
    beta = 1.0
    agents = [string(i) for i in -5:5]
    true_skills = [i for i in -5.0:5.0] 
    mu_prior = [ abs(i)==5.0 ? i : 0.0 for i in -5.0:5.0]
    sigma_prior = [ abs(i)==5.0 ? 1e-7 : 6.0 for i in -5.0:5.0]
    priors = Dict{String,ttt.Player}()
    for i in -5:5
        priors[string(i)] = ttt.Player(ttt.Gaussian(mu_prior[i+6], sigma_prior[i+6]), beta, 0.0) 
    end
    
    events = [ [[string(a-1)],[string(a)]]  for e in 1:100 for a in -4:5]
    results = [ (Random.randn(1)[1]*beta + a-1) > (Random.randn(1)[1]*beta + a) ? [0.,1.] : [1.,0.] for i in 1:100 for a in -4.0:5.0] 
    times = [ e  for e in 1:100 for a in -4:5]
    h = ttt.History(events , results, times, priors, iter=100)
    ttt.convergence(h)
    diffs = [(ttt.posterior(h.batches[1],string(a))-ttt.posterior(h.batches[1],string(a-1))).mu for a in -3:4]
    probs = [ 1-ttt.cdf(ttt.Gaussian(d,sqrt(2)),0.0)  for d in diffs]
    for p in probs 
        @test abs(1-ttt.cdf(ttt.Gaussian(1.0,sqrt(2)),0.0)-p) < 0.04
    end
end
@testset "Online predictions" begin
    priors = Dict{String,ttt.Player}()
    priors["a"] = ttt.Player(ttt.Gaussian(0.0, 3.0), 1.0, 0.0) 
    priors["b"] = ttt.Player(ttt.Gaussian(0.0, 3.0), 1.0, 0.0) 
    priors["c"] = ttt.Player(ttt.Gaussian(2.0, 0.5), 1.0, 0.0) 
#     
    composition = [[["a"],["b"]], [["a"],["c"]], [["a"],["b"]]]
    results = [ [0.,1.], [1.,0.], [1.,0.]]
    times = [1,2,3]
    h = ttt.History(composition , results, times, priors, iter=100, gamma=0.0,online=true)
    
    h.batches[3].skills["b"].forward
    h.batches[3].skills["b"].online
end
end

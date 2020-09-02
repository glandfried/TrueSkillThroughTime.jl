include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test
using CSV
using DataFrames
using JLD2

@testset "Tests" begin
    @testset "ppf" begin
        @test isapprox(ttt.ppf(ttt.N01,0.3),-0.52440044)
        @test isapprox(ttt.ppf(ttt.Gaussian(2.,3.),0.3),0.42679866)
    end
    @testset "cdf" begin
        @test isapprox(ttt.cdf(ttt.N01,0.3),0.617911409)
        @test isapprox(ttt.cdf(ttt.Gaussian(2.,3.),0.3),0.28547031)
    end
    @testset "pdf" begin
        @test isapprox(ttt.pdf(ttt.N01,0.3),0.38138781)
        @test isapprox(ttt.pdf(ttt.Gaussian(2.,3.),0.3),0.11325579)
    end
    @testset "compute_margin" begin
        @test isapprox(ttt.compute_margin(0.25,2),1.8776005988)
        @test isapprox(ttt.compute_margin(0.25,3),2.29958170)
        @test isapprox(ttt.compute_margin(0.0,3),2.7134875810435737e-07)
        @test isapprox(ttt.compute_margin(1.0,3),Inf)
    end
    @testset "trunc" begin
        res = ttt.trunc(ttt.N01,0.0,false)
        @test isapprox(res,ttt.Gaussian(0.79788453,0.602810306),1e-5) 
        margin = 1.8776005988
        res = ttt.trunc(ttt.Gaussian(0.,sqrt(2)*(25/6) ),margin,true)
        @test isapprox(res, ttt.Gaussian(0.,1.076707),1e-5)
        res = ttt.trunc(ttt.Gaussian(12.,sqrt(2)*(25/6) ),margin,true)
        @test isapprox(res, ttt.Gaussian(0.3900999,1.034401),1e-5)
    end
    @testset "Gaussian" begin
        N, M = ttt.Gaussian(), ttt.Gaussian(0.0, 1.0)
        @test isapprox(M/N, ttt.Gaussian(-0.365, 1.007),1e-3)
        @test isapprox(N*M, ttt.Gaussian(0.355, 0.993),1e-3)
        @test isapprox(N+M, ttt.Gaussian(25.00, 8.393),1e-3)
        M = ttt.Gaussian(1.0, 1.0)
        @test isapprox(N-M, ttt.Gaussian(24.00, 8.393),1e-3)
    end
    @testset "1vs1" begin
        ta = [ttt.Rating()]
        tb = [ttt.Rating()]
        g = ttt.Game([ta,tb], [1,0])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(20.794779,7.194481), 1e-4) 
        @test isapprox(post[2][1], ttt.Gaussian(29.205220,7.194481), 1e-4)
        
        g = ttt.Game([[ttt.Rating(29.,1.)] ,[ttt.Rating()]], [1,0], 0.0)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(28.896,0.996), 1e-3) 
        @test isapprox(post[2][1], ttt.Gaussian(32.189,6.062), 1e-3)
    end
    @testset "1vs1vs1" begin
        g = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000000,6.238469796),1e-5)
        @test isapprox(post[2][1],ttt.Gaussian(31.3113582213,6.69881865) ,1e-5)
        
        g = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2], 0.5)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.00000,6.48760),1e-5)
        @test isapprox(post[2][1],ttt.Gaussian(29.19950,7.00947),1e-5)
        @test isapprox(post[3][1],ttt.Gaussian(20.80049,7.00947),1e-5)
    end
    @testset "1vs1 draw" begin
        ta = [ttt.Rating()]
        tb = [ttt.Rating()]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000,6.469),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.000,6.469),1e-3)
        
        ta = [ttt.Rating(25.,3.)]
        tb = [ttt.Rating(29.,2.)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.736,2.710),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(28.672,1.916),1e-3)
    end
    @testset "1vs1vs1 draw" begin
        ta = [ttt.Rating()]
        tb = [ttt.Rating()]
        tc = [ttt.Rating()]
        g = ttt.Game([ta,tb,tc], [0,0,0],0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000,5.746947),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.000,5.714755),1e-3)
        
        ta = [ttt.Rating(25.,3.)]
        tb = [ttt.Rating(25.,3.)]
        tc = [ttt.Rating(29.,2.)]
        g = ttt.Game([ta,tb,tc], [0,0,0],0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.473,2.645),1e-2)
        @test isapprox(post[2][1],ttt.Gaussian(25.505,2.631),1e-2)
        @test isapprox(post[3][1],ttt.Gaussian(28.565,1.888),1e-2)
    end
    @testset "NvsN Draw" begin
        ta = [ttt.Rating(15.,1.),ttt.Rating(15.,1.)]
        tb = [ttt.Rating(30.,2.)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[1][2],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(30.000,1.9320),1e-3)
    end
    @testset "Game evidence" begin
        
        @testset "1vs1" begin
            ta = [ttt.Rating(25.,1e-7)]
            tb = [ttt.Rating(25.,1e-7)]
            g = ttt.Game([ta,tb], [0,0], 0.25)
            @test isapprox(g.evidence,0.25)
            g = ttt.Game([ta,tb], [0,1], 0.25)
            @test isapprox(g.evidence,0.375)
        end
        @testset "1vs1vs1 margin 0" begin
            ta = [ttt.Rating(25.,1e-7)]
            tb = [ttt.Rating(25.,1e-7)]
            tc = [ttt.Rating(25.,1e-7)]
            
            
            g_abc = ttt.Game([ta,tb,tc], [1,2,3], 0.)
            g_acb = ttt.Game([ta,tb,tc], [1,3,2], 0.)
            g_bac = ttt.Game([ta,tb,tc], [2,1,3], 0.)
            g_bca = ttt.Game([ta,tb,tc], [3,1,2], 0.)
            g_cab = ttt.Game([ta,tb,tc], [2,3,1], 0.)
            g_cba = ttt.Game([ta,tb,tc], [3,2,1], 0.)
            
            d1 = ttt.performance(g_abc,1)-ttt.performance(g_abc,2)
            
            proba = 0
            proba += g_abc.evidence
            proba += g_acb.evidence
            proba += g_bac.evidence
            proba += g_bca.evidence
            proba += g_cab.evidence
            proba += g_cba.evidence            
            println("Corregir la evidencia multiequipos para que sume 1")
            @test  isapprox(proba, 1.49999991)
        end
    end
    @testset "Forget" begin
        r = ttt.Rating(25.,1e-7)
        @test isapprox(ttt.forget(r,5).N.sigma,6.25)
        @test isapprox(ttt.forget(r,1).N.sigma,1.25)
    end
    @testset "Batch" begin
        @testset "One event each" begin
            b = ttt.Batch([ [["a"],["b"]], [["c"],["d"]] , [["e"],["f"]] ], [[0,1],[1,0],[0,1]], 2)
            @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(29.205,7.194),1e-3)
            @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(20.795,7.194),1e-3)
            @test isapprox(ttt.posterior(b,"d"),ttt.Gaussian(29.205,7.194),1e-3)
            @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(20.795,7.194),1e-3)
            @test isapprox(ttt.posterior(b,"e"),ttt.Gaussian(29.205,7.194),1e-3)
            @test isapprox(ttt.posterior(b,"f"),ttt.Gaussian(20.795,7.194),1e-3)
            iter = ttt.convergence(b)
            @test iter == 0
        end
        @testset "Same strength" begin
            b = ttt.Batch([ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ], [[0,1],[1,0],[0,1]], 2)
            @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(24.96097,6.29954),1e-3)
            @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(27.09559,6.01033),1e-3)
            @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(24.88968,5.86631),1e-3)
            iter = ttt.convergence(b)
            @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(25.000,5.419),1e-3)
            @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(25.000,5.419),1e-3)
            @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(25.000,5.419),1e-3)
        end
    end
    @testset "History" begin
        @testset "TrueSkill initalization" begin
            events = [ [["aa"],["b"]], [["aa"],["c"]] , [["b"],["c"]] ]
            results = [[0,1],[1,0],[0,1]]
            h = ttt.History(events, results, [1,2,3])

            
            @test !(h.batches[1].max_step > 1e-6) & !(h.batches[2].max_step > 1e-6)
            @test isapprox(ttt.posterior(h.batches[1],"aa"),ttt.Gaussian(29.205,7.19448),1e-3)

            observed = h.batches[2].prior_forward["aa"].N.sigma 
            expected = sqrt((ttt.GAMMA*1)^2 +  ttt.posterior(h.batches[1],"aa").sigma^2)
            @test isapprox(observed, expected)
            
            observed = ttt.posterior(h.batches[2],"aa")
            g = ttt.Game([[h.batches[2].prior_forward["aa"]],[h.batches[2].prior_forward["c"]]],[1,0])
            expected = ttt.posteriors(g)[1][1]
            @test isapprox(observed, expected, 1e-7)
        end
        @testset "One-batch history" begin
            composition = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
            results = [[0,1],[0,1],[0,1]]
            bache = [1,1,1]
            h1 = ttt.History(composition,results, bache)
            # TrueSkill
            @test isapprox(ttt.posterior(h1.batches[1],"aj"),ttt.Gaussian(22.904,6.010),2)
            @test isapprox(ttt.posterior(h1.batches[1],"cj"),ttt.Gaussian(25.110,5.866),2)
            # TTT
            step , i = ttt.convergence(h1)
            @test isapprox(ttt.posterior(h1.batches[1],"aj"),ttt.Gaussian(25.000,5.419),2)
            @test isapprox(ttt.posterior(h1.batches[1],"cj"),ttt.Gaussian(25.000,5.419),2)
            
            h2 = ttt.History(composition,results, [1,2,3])
            # TrueSkill
            @test isapprox(ttt.posterior(h2.batches[3],"aj"),ttt.Gaussian(22.904,6.012),2)
            @test isapprox(ttt.posterior(h2.batches[3],"cj"),ttt.Gaussian(25.110,5.867),2)
            # TTT
            step , i = ttt.convergence(h2)
            @test isapprox(ttt.posterior(h2.batches[3],"aj"),ttt.Gaussian(24.997,5.421),2)
            @test isapprox(ttt.posterior(h2.batches[3],"cj"),ttt.Gaussian(25.000,5.420),2)
        end
        @testset "TrueSkill Through Time" begin
            events = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
            results = [[0,1],[1,0],[0,1]]
            h = ttt.History(events, results)
            step , iter = ttt.convergence(h)
            @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(25.0002673,5.41950697),1e-5)
            @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(24.9986633,5.41968377),1e-5)
            @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian(25.0029304,5.42076739),1e-5)            
            
        end
        @testset "Learning curves" begin
            events = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
            results = [[0,1],[0,1],[0,1]]    
            h = ttt.History(events, results, [5,6,7])
            ttt.convergence(h)
            lc = ttt.learning_curves(h)
            
            @test lc["aj"][1][1] == 5
            @test lc["aj"][end][1] == 7
            @test isapprox(lc["aj"][end][2],ttt.Gaussian(24.997,5.421),2)
            @test isapprox(lc["cj"][end][2],ttt.Gaussian(25.000,5.420),2)
        end
    end
    @testset "CSV Same strength" begin 
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
    
        CSV.write("same_strength.csv", df; header=true)
        @test true
    end
    @testset "CSV Same strength two groups" begin 
        events = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ,[["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] 
                  ,[["ai"],["bi"]],[["bi"],["ci"]], [["ci"],["ai"]] ,[["ai"],["bi"]],[["bi"],["ci"]], [["ci"],["ai"]]
                  , [["aj"],["ai"]], [["aj"],["ai"]], [["aj"],["ai"]], [["aj"],["ai"]], [["aj"],["ai"]], [["aj"],["ai"]]
                  , [["aj"],["ai"]], [["aj"],["ai"]] ]
        results = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]
                  ,[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]
                  , [0,1],[0,1],[1,0],[0,1],[0,1],[1,0],[0,1],[0,1]]    
        priors = Dict{String,ttt.Rating}()
        for k in ["aj", "bj", "cj", "ai", "bi", "ci"]
            priors[k] = ttt.Rating(0., 3.0, 0.5, 0.0, k ) 
        end
        h = ttt.History(events, results, [1,2,3,4,5,6, 1,2,3,4,5,6, 7,7,7,7,7,7,7,7], priors)
        trueSkill_evidence = ttt.log_evidence(h)
        lc_ts = ttt.learning_curves(h)
        ttt.convergence(h)
        throughTime = ttt.log_evidence(h)
        evs = [e for b in h.batches for e in b.evidences]
        lc_ttt = ttt.learning_curves(h)
        
        # Predicciones
        Nbeta = ttt.Gaussian(0.0,0.5)
        1-ttt.cdf(lc_ttt["aj"][1][2]+Nbeta -lc_ttt["bj"][1][2]+Nbeta,0.) 
        1-ttt.cdf(lc_ttt["cj"][1][2]+Nbeta -lc_ttt["bj"][1][2]+Nbeta,0.) 
        1-ttt.cdf(lc_ttt["ai"][1][2]+Nbeta -lc_ttt["bi"][1][2]+Nbeta,0.) 
        1-ttt.cdf(lc_ttt["ci"][1][2]+Nbeta -lc_ttt["bi"][1][2]+Nbeta,0.) 
        1-ttt.cdf(lc_ttt["bj"][1][2]+Nbeta -lc_ttt["bi"][1][2]+Nbeta,0.) 
        1-ttt.cdf(lc_ttt["aj"][1][2]+Nbeta -lc_ttt["ai"][1][2]+Nbeta,0.) 
        
        # Diferencia media
        #1-ttt.cdf(ttt.Gaussian(15.0,sqrt(2*15.0^2)),0.) 
        1-ttt.cdf(ttt.Gaussian((lc_ttt["aj"][1][2]-lc_ttt["ai"][1][2]).mu,sqrt(2*0.5^2)),0.) 
        
        1-ttt.cdf(ttt.Gaussian(0.33725,sqrt(2*0.5^2) ),0.) 
        
        
        df = DataFrame(mu_a_ts = [ N.mu for (k,N) in lc_ts["aj"] if k != 7]
                      ,sigma_a_ts = [ N.sigma for (k,N) in lc_ts["aj"]  if k != 7]
                      ,mu_a_ttt = [ N.mu for (k,N) in lc_ttt["aj"]  if k != 7]
                      ,sigma_a_ttt = [ N.sigma for (k,N) in  lc_ttt["aj"]  if k != 7]
                      ,time_a = [ k for (k,N) in lc_ts["aj"]  if k != 7]
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
                      ,mu_ai_ts = [ N.mu for (k,N) in lc_ts["ai"] if k != 7]
                      ,sigma_ai_ts = [ N.sigma for (k,N) in lc_ts["ai"]  if k != 7]
                      ,mu_ai_ttt = [ N.mu for (k,N) in lc_ttt["ai"]  if k != 7]
                      ,sigma_ai_ttt = [ N.sigma for (k,N) in  lc_ttt["ai"]  if k != 7]
                      ,time_ai = [ k for (k,N) in lc_ts["ai"]  if k != 7]
                      ,mu_bi_ts = [ N.mu for (k,N) in lc_ts["bi"]]
                      ,sigma_bi_ts = [ N.sigma for (k,N) in lc_ts["bi"]]
                      ,mu_bi_ttt = [ N.mu for (k,N) in lc_ttt["bi"]]
                      ,sigma_bi_ttt = [ N.sigma for (k,N) in lc_ttt["bi"]]
                      ,time_bi = [ k for (k,N) in lc_ts["bi"]]
                      ,mu_ci_ts = [ N.mu for (k,N) in lc_ts["ci"]]
                      ,sigma_ci_ts = [ N.sigma for (k,N) in lc_ts["ci"]]
                      ,mu_ci_ttt = [ N.mu for (k,N) in lc_ttt["ci"]]
                      ,sigma_ci_ttt = [ N.sigma for (k,N) in lc_ttt["ci"]]
                      ,time_ci = [ k for (k,N) in lc_ts["ci"]]
                      )
    
        CSV.write("same_strength_two_groups.csv", df; header=true)
        @test true
    end
    @testset "Write smoothing messages"
        events = [ [["a"],["b"]], [["a"],["b"]]]
        results = [[0,1],[1,0]]
        times = [1,2]
        priors = Dict{String,ttt.Rating}()
        for k in ["a", "b"]
            priors[k] = ttt.Rating(0., 3.0, 0.5, 0.0, k ) 
        end
        h = ttt.History(events, results, times, priors)
        fp_a = Vector{ttt.Gaussian}()
        bp_a = Vector{ttt.Gaussian}() 
        lh_a = Vector{ttt.Gaussian}()
        wp_a = Vector{ttt.Gaussian}()
        wp_b = Vector{ttt.Gaussian}()
        p_a =  Vector{ttt.Gaussian}()
        
        push!(fp_a, h.batches[1].prior_forward["a"].N)
        push!(bp_a, h.batches[1].prior_backward["a"])
        push!(lh_a, h.batches[1].likelihood["a"][1])
        push!(wp_a, fp_a[end]*bp_a[end])
        push!(p_a, wp_a[end]*lh_a[end])
        push!(wp_b, h.batches[1].prior_forward["b"].N*h.batches[1].prior_backward["b"])
        for _ in 1:10
            ttt.iteration(h)
            push!(fp_a, h.batches[1].prior_forward["a"].N)
            push!(bp_a, h.batches[1].prior_backward["a"])
            push!(lh_a, h.batches[1].likelihood["a"][1])
            push!(wp_a, fp_a[end]*bp_a[end])
            push!(p_a, wp_a[end]*lh_a[end])
            push!(wp_b, h.batches[1].prior_forward["b"].N*h.batches[1].prior_backward["b"])
        end
        
        df = DataFrame(fp_mu = [N.mu for N in fp_a]
                      ,fp_sigma = [N.sigma for N in fp_a]
                      ,bp_mu = [N.mu for N in bp_a]
                      ,bp_sigma = [N.sigma for N in bp_a]
                      ,lh_mu = [N.mu for N in lh_a]
                      ,lh_sigma = [N.sigma for N in lh_a]
                      ,wp_mu = [N.mu for N in wp_a]
                      ,wp_sigma = [N.sigma for N in wp_a]
                      ,p_mu = [N.mu for N in p_a]
                      ,p_sigma = [N.sigma for N in p_a]
                      ,wp_opponent_mu = [N.mu for N in wp_b]
                      ,wp_opponent_sigma = [N.sigma for N in wp_b])
        CSV.write("smoothing.csv", df; header=true)
        
    end
    @testset "TTT vs TrueSkill evidence"
        
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
            
        CSV.write("ttt_vs_ts.csv", df; header=true)
 
    end
        
end


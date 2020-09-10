include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test

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
        
        #@test isapprox(ttt.compute_margin(0.25,sqrt(2)*25.0/6),1.8776005988)
        #@test isapprox(ttt.compute_margin(0.25,sqrt(3)*25.0/6),2.29958170)
        #@test isapprox(ttt.compute_margin(0.0,sqrt(3)*25.0/6),2.7134875810435737e-07)
        #@test isapprox(ttt.compute_margin(1.0,sqrt(3)*25.0/6),Inf)
    end
    @testset "trunc" begin
        res = ttt.trunc(ttt.N01,0.0,false)
        @test isapprox(res,ttt.Gaussian(0.79788453,0.602810306),1e-5) 
        margin = 1.8776005988
        res = ttt.trunc(ttt.Gaussian(0.,sqrt(2)*(25/6) ),margin,false)
        @test isapprox(res, ttt.Gaussian(5.958, 3.226), 1e-3)
        res = ttt.trunc(ttt.Gaussian(0.,sqrt(2)*(25/6) ),margin,true)
        @test isapprox(res, ttt.Gaussian(0.,1.076707),1e-4)
        res = ttt.trunc(ttt.Gaussian(12.,sqrt(2)*(25/6) ),margin,true)
        @test isapprox(res, ttt.Gaussian(0.3900999,1.034401),1e-5)
    end
    @testset "Gaussian" begin
        N, M = ttt.Gaussian(25.0, 25.0/3), ttt.Gaussian(0.0, 1.0)
        @test isapprox(M/N, ttt.Gaussian(-0.365, 1.007),1e-3)
        @test isapprox(N*M, ttt.Gaussian(0.355, 0.993),1e-3)
        @test isapprox(N+M, ttt.Gaussian(25.00, 8.393),1e-3)
        M = ttt.Gaussian(1.0, 1.0)
        @test isapprox(N-M, ttt.Gaussian(24.00, 8.393),1e-3)
    end
    @testset "1vs1" begin
        ta = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        tb = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [1,0], 0.0)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(20.794779,7.194481), 1e-4) 
        @test isapprox(post[2][1], ttt.Gaussian(29.205220,7.194481), 1e-4)
        
        g = ttt.Game([[ttt.Rating(29.,1.)] ,[ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]], [1,0])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(28.896,0.996), 1e-3) 
        @test isapprox(post[2][1], ttt.Gaussian(32.189,6.062), 1e-3)
    end
    @testset "1vs1vs1" begin
        g = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000000,6.238469796),1e-5)
        @test isapprox(post[2][1],ttt.Gaussian(31.3113582213,6.69881865) ,1e-5)
        
        g = ttt.Game([[ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)],[ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)],[ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]], [1,0,2], 0.5)
        post = ttt.posteriors(g)
        
        @test isapprox(post[1][1],ttt.Gaussian(25.000,6.093),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(33.379,6.484),1e-3)
        @test isapprox(post[3][1],ttt.Gaussian(16.621,6.484),1e-3)
    end
    @testset "1vs1 draw" begin
        ta = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        tb = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000,6.469),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.000,6.469),1e-3)
        
        ta = [ttt.Rating(25.,3.,25.0/6,25.0/300)]
        tb = [ttt.Rating(29.,2.,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.736,2.710),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(28.672,1.916),1e-3)
    end
    @testset "1vs1vs1 draw" begin
        ta = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        tb = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        tc = [ttt.Rating(25.0,25.0/3,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb,tc], [0,0,0],0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000,5.729),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.000,5.707),1e-3)
        
        ta = [ttt.Rating(25.,3.,25.0/6,25.0/300)]
        tb = [ttt.Rating(25.,3.,25.0/6,25.0/300)]
        tc = [ttt.Rating(29.,2.,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb,tc], [0,0,0],0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.489,2.638),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.511,2.629),1e-3)
        @test isapprox(post[3][1],ttt.Gaussian(28.556,1.886),1e-3)
    end
    @testset "NvsN Draw" begin
        ta = [ttt.Rating(15.,1.,25.0/6,25.0/300)
             ,ttt.Rating(15.,1.,25.0/6,25.0/300)]
        tb = [ttt.Rating(30.,2.,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[1][2],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(30.000,1.9320),1e-3)
    end
    @testset "NvsNvsN mixt" begin
        ta = [ttt.Rating(12.,3.,25.0/6,25.0/300)
             ,ttt.Rating(18.,3.,25.0/6,25.0/300)]
        tb = [ttt.Rating(30.,3.,25.0/6,25.0/300)]
        tc = [ttt.Rating(14.,3.,25.0/6,25.0/300)
             ,ttt.Rating(16.,3.,25.0/6,25.0/300)]
        g = ttt.Game([ta,tb, tc], [0,1,1], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(13.051,2.864),1e-3)
        @test isapprox(post[1][2],ttt.Gaussian(19.051,2.864),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(29.292,2.764),1e-3)
        @test isapprox(post[3][1],ttt.Gaussian(13.658,2.813),1e-3)
        @test isapprox(post[3][2],ttt.Gaussian(15.658,2.813),1e-3)
    end
    @testset "Game evidence" begin
        @testset "1vs1" begin
            ta = [ttt.Rating(25.,1e-7,25.0/6,25.0/300)]
            tb = [ttt.Rating(25.,1e-7,25.0/6,25.0/300)]
            g = ttt.Game([ta,tb], [0,0], 0.25)
            @test isapprox(g.evidence,0.25)
            g = ttt.Game([ta,tb], [0,1], 0.25)
            @test isapprox(g.evidence,0.375)
        end
        @testset "1vs1vs1 margin 0" begin
            ta = [ttt.Rating(25.,1e-7,25.0/6,25.0/300)]
            tb = [ttt.Rating(25.,1e-7,25.0/6,25.0/300)]
            tc = [ttt.Rating(25.,1e-7,25.0/6,25.0/300)]
            
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
        r = ttt.Rating(25.,1e-7,25.0/6,0.15*25.0/3)
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
            @test isapprox(ttt.posterior(h1.batches[1],"aj"),ttt.Gaussian(22.904,6.010),1e-3)
            @test isapprox(ttt.posterior(h1.batches[1],"cj"),ttt.Gaussian(25.110,5.866),1e-3)
            # TTT
            step , i = ttt.convergence(h1)
            @test isapprox(ttt.posterior(h1.batches[1],"aj"),ttt.Gaussian(25.000,5.419),1e-3)
            @test isapprox(ttt.posterior(h1.batches[1],"cj"),ttt.Gaussian(25.000,5.419),1e-3)
            
            priors = Dict{String,ttt.Rating}()
            for k in ["aj", "bj", "cj"]
                priors[k] = ttt.Rating(25., 25.0/3, 25.0/6, 25.0/300, k ) 
            end
            h2 = ttt.History(composition,results, [1,2,3], priors  )
            # TrueSkill
            @test isapprox(ttt.posterior(h2.batches[3],"aj"),ttt.Gaussian(22.902,6.012),1e-3)
            @test isapprox(ttt.posterior(h2.batches[3],"cj"),ttt.Gaussian(25.110,5.867),1e-3)
            # TTT
            step , i = ttt.convergence(h2)
            @test isapprox(ttt.posterior(h2.batches[3],"aj"),ttt.Gaussian(24.997,5.421),1e-3)
            @test isapprox(ttt.posterior(h2.batches[3],"cj"),ttt.Gaussian(25.000,5.420),1e-3)
        end
        @testset "TrueSkill Through Time" begin
            events = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
            results = [[0,1],[1,0],[0,1]]
            priors = Dict{String,ttt.Rating}()
            for k in ["a", "b", "c"]
                priors[k] = ttt.Rating(25., 25.0/3, 25.0/6, 25.0/300, k ) 
            end
            
            h = ttt.History(events, results, Int64[], priors)
            step , iter = ttt.convergence(h)
            @test (h.batches[3].elapsed["b"] == 1) & (h.batches[3].elapsed["c"] == 1)
            @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(25.0002673,5.41938162),1e-5)
            @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(24.999465,5.419425831),1e-5)
            @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian(25.00053219,5.419696790),1e-5)
            #ttt.learning_curves(h)
        end
        @testset "Learning curves" begin
            events = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
            results = [[0,1],[0,1],[0,1]]    
            priors = Dict{String,ttt.Rating}()
            for k in ["aj", "bj", "cj"]
                priors[k] = ttt.Rating(25., 25.0/3, 25.0/6, 25.0/300, k ) 
            end
            h = ttt.History(events, results, [5,6,7], priors)
            ttt.convergence(h)
            lc = ttt.learning_curves(h)
            
            @test lc["aj"][1][1] == 5
            @test lc["aj"][end][1] == 7
            @test isapprox(lc["aj"][end][2],ttt.Gaussian(24.997,5.421),1e-3)
            @test isapprox(lc["cj"][end][2],ttt.Gaussian(25.000,5.420),13-3)
        end
    end        
end


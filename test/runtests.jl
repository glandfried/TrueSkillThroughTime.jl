include("../src/TrueSkillThroughTime.jl")
using .TrueSkillThroughTime
global const ttt = TrueSkillThroughTime
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
        @test isapprox(ttt.compute_margin(0.25,sqrt(2)*25.0/6),1.8776005988)
        @test isapprox(ttt.compute_margin(0.25,sqrt(3)*25.0/6),2.29958170)
        @test isapprox(ttt.compute_margin(0.0,sqrt(3)*25.0/6),2.7134875810435737e-07)
        @test isapprox(ttt.compute_margin(1.0,sqrt(3)*25.0/6),Inf)
    end
    @testset "approx trunc" begin
        res = ttt.approx(ttt.N01,0.0,false)
        @test isapprox(res,ttt.Gaussian(0.79788453,0.602810306),1e-5) 
        margin = 1.8776005988
        res = ttt.approx(ttt.Gaussian(0.,sqrt(2)*(25/6) ),margin,false)
        @test isapprox(res, ttt.Gaussian(5.958, 3.226), 1e-3)
        res = ttt.approx(ttt.Gaussian(0.,sqrt(2)*(25/6) ),margin,true)
        @test isapprox(res, ttt.Gaussian(0.,1.0767055),1e-6)
        res = ttt.approx(ttt.Gaussian(12.,sqrt(2)*(25/6) ),margin,true)
        @test isapprox(res, ttt.Gaussian(0.3900995,1.0343979),1e-6)
    res.sigma
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
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0.,1.], 0.0)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(20.794779,7.194481), 1e-4) 
        @test isapprox(post[2][1], ttt.Gaussian(29.205220,7.194481), 1e-4)
        
        g = ttt.Game([[ttt.Player(ttt.Gaussian(29.,1.),25.0/6)] ,[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6)]], [0.,1.])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(28.896,0.996), 1e-3) 
        @test isapprox(post[2][1], ttt.Gaussian(32.189,6.062), 1e-3)
        
        ta = [ttt.Player(ttt.Gaussian(1.139,0.531),1.0,0.2125)]
        tb = [ttt.Player(ttt.Gaussian(15.568,0.51),1.0,0.2125)]
        g = ttt.Game([ta,tb], [0.,1.], 0.0)
        @test isapprox(g.likelihoods[1][1].sigma, Inf)
        @test isapprox(g.likelihoods[1][1].mu, 0.0)
        @test isapprox(g.likelihoods[2][1].sigma, Inf)
        
    end
    @testset "1vs1vs1" begin
        teams = [[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
                ,[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
                ,[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]]
    
        g = ttt.Game(teams, [1.,2.,0.])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000000,6.238469796),1e-5)
        @test isapprox(post[2][1],ttt.Gaussian(31.3113582213,6.69881865) ,1e-5)
        
        g = ttt.Game(teams)
        post = ttt.posteriors(g)
        @test isapprox(post[2][1],ttt.Gaussian(25.000000,6.238469796),1e-5)
        @test isapprox(post[1][1],ttt.Gaussian(31.3113582213,6.69881865) ,1e-5)
        
        g = ttt.Game(teams, [1.,2.,0.], 0.5)
        post = ttt.posteriors(g)
        
        @test isapprox(post[1][1],ttt.Gaussian(25.000,6.093),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(33.379,6.484),1e-3)
        @test isapprox(post[3][1],ttt.Gaussian(16.621,6.484),1e-3)
    end
    @testset "1vs1 draw" begin
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0.,0.], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000,6.469481),1e-4)
        @test isapprox(post[2][1],ttt.Gaussian(25.000,6.469481),1e-4)
        
        ta = [ttt.Player(ttt.Gaussian(25.,3.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(29.,2.),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0.,0.], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.736,2.709956),1e-4)
        @test isapprox(post[2][1],ttt.Gaussian(28.67289,1.916471),1e-4)
    end
    @testset "1vs1vs1 draw" begin
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        tc = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb,tc], [0.,0.,0.],0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.000,5.729),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.000,5.707),1e-3)
        
        ta = [ttt.Player(ttt.Gaussian(25.,3.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.,3.),25.0/6,25.0/300)]
        tc = [ttt.Player(ttt.Gaussian(29.,2.),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb,tc], [0.,0.,0.],0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(25.489,2.638),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(25.511,2.629),1e-3)
        @test isapprox(post[3][1],ttt.Gaussian(28.556,1.886),1e-3)
    end
    @testset "NvsN Draw" begin
        ta = [ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300)
             ,ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(30.,2.),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0.,0.], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[1][2],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(30.000,1.9320),1e-3)
    end
    @testset "NvsNvsN mixt" begin
        ta = [ttt.Player(ttt.Gaussian(12.,3.),25.0/6,25.0/300)
             ,ttt.Player(ttt.Gaussian(18.,3.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(30.,3.),25.0/6,25.0/300)]
        tc = [ttt.Player(ttt.Gaussian(14.,3.),25.0/6,25.0/300)
             ,ttt.Player(ttt.Gaussian(16.,3.),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb, tc], [1.,0.,0.], 0.25)
        post = ttt.posteriors(g)
        @test isapprox(post[1][1],ttt.Gaussian(13.051,2.864),1e-3)
        @test isapprox(post[1][2],ttt.Gaussian(19.051,2.864),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(29.292,2.764),1e-3)
        @test isapprox(post[3][1],ttt.Gaussian(13.658,2.813),1e-3)
        @test isapprox(post[3][2],ttt.Gaussian(15.658,2.813),1e-3)
    end
    @testset "Game evidence" begin
        @testset "1vs1" begin
            ta = [ttt.Player(ttt.Gaussian(27.,2.),1.0)]
            tb = [ttt.Player(ttt.Gaussian(25.,3.0),1.0)]
            d = ta[1].prior.mu - tb[1].prior.mu
            v = sqrt(2+ta[1].prior.sigma^2 + tb[1].prior.sigma^2)
            evidence = 1-ttt.cdf(ttt.Gaussian(d,v),0.0)
            g = ttt.Game([ta,tb], [1.,0.])
            @test isapprox(g.evidence , evidence)
            
            ta = [ttt.Player(ttt.Gaussian(27.,2.),1.0),ttt.Player(ttt.Gaussian(28.,1.),1.0) ]
            tb = [ttt.Player(ttt.Gaussian(25.,3.0),1.0),ttt.Player(ttt.Gaussian(26.,1.0),1.0)]
            d = (ta[1].prior.mu + ta[2].prior.mu) - (tb[1].prior.mu + tb[2].prior.mu)
            v = sqrt(4+ta[1].prior.sigma^2 + ta[2].prior.sigma^2+ tb[1].prior.sigma^2 + tb[2].prior.sigma^2 )
            evidence = 1-ttt.cdf(ttt.Gaussian(d,v),0.0)
            g = ttt.Game([ta,tb], [1.,0.])
            @test isapprox(g.evidence , evidence)
            
            ta = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
            tb = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
            g = ttt.Game([ta,tb], [0.,0.], 0.25)
            @test isapprox(g.evidence,0.25)
            g = ttt.Game([ta,tb], [1.,0.], 0.25)
            @test isapprox(g.evidence,0.375)
        end
        @testset "1vs1vs1 margin 0" begin
            ta = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
            tb = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
            tc = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
            
            g_abc = ttt.Game([ta,tb,tc], [3.,2.,1.], 0.)
            g_acb = ttt.Game([ta,tb,tc], [3.,1.,2.], 0.)
            g_bac = ttt.Game([ta,tb,tc], [2.,3.,1.], 0.)
            g_bca = ttt.Game([ta,tb,tc], [1.,3.,2.], 0.)
            g_cab = ttt.Game([ta,tb,tc], [2.,1.,3.], 0.)
            g_cba = ttt.Game([ta,tb,tc], [1.,2.,3.], 0.)
            
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
        gamma = 0.15*25.0/3
        g = ttt.Gaussian(25.,1e-7)
        @test isapprox(ttt.forget(g,gamma,5).sigma,sqrt(5*gamma^2))
        @test isapprox(ttt.forget(g,gamma,1).sigma,sqrt(1*gamma^2))
    end
    @testset "One event each" begin
        agents = Dict{String,ttt.Agent}()
        for k in ["a", "b", "c", "d", "e", "f"]
            agents[k] = ttt.Agent(ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300 ) , ttt.Ninf, ttt.minInt64)
        end
        composition = [ [["a"],["b"]], [["c"],["d"]] , [["e"],["f"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        b = ttt.Batch(composition = composition, results = results, agents = agents)
        @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(29.205,7.194),1e-3)
        @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(20.795,7.194),1e-3)
        @test isapprox(ttt.posterior(b,"d"),ttt.Gaussian(29.205,7.194),1e-3)
        @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(20.795,7.194),1e-3)
        @test isapprox(ttt.posterior(b,"e"),ttt.Gaussian(29.205,7.194),1e-3)
        @test isapprox(ttt.posterior(b,"f"),ttt.Gaussian(20.795,7.194),1e-3)
        iter = ttt.convergence(b)
        @test iter == 1
    end
    @testset "Same strength" begin
        agents= Dict{String,ttt.Agent}()
        for k in ["a", "b", "c", "d", "e", "f"]
            agents[k] = ttt.Agent(ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300 ) , ttt.Ninf, ttt.minInt64)
        end
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        b = ttt.Batch(composition = composition, results = results, agents = agents)
        @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(24.96097,6.29954),1e-3)
        @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(27.09559,6.01033),1e-3)
        @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(24.88968,5.86631),1e-3)
        iter = ttt.convergence(b)
        @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(25.000,5.419),1e-3)
        @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(25.000,5.419),1e-3)
        @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(25.000,5.419),1e-3)
    end
    @testset "Add events into a batch" begin
        agents= Dict{String,ttt.Agent}()
        for k in ["a", "b", "c", "d", "e", "f"]
            agents[k] = ttt.Agent(ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300 ) , ttt.Ninf, ttt.minInt64)
        end
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        b = ttt.Batch(composition = composition, results = results, agents = agents)
        iter = ttt.convergence(b)
        @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(25.000,5.419),1e-3)
        @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(25.000,5.419),1e-3)
        @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(25.000,5.419),1e-3)
        ttt.add_events(b,composition,results)
        @test length(b.events) == 6 
        iter = ttt.convergence(b,1e-6,20)
        @test iter == 20
        @test isapprox(ttt.posterior(b,"a"),ttt.Gaussian(25.000,3.88),1e-3)
        @test isapprox(ttt.posterior(b,"b"),ttt.Gaussian(25.000,3.88),1e-3)
        @test isapprox(ttt.posterior(b,"c"),ttt.Gaussian(25.000,3.88),1e-3)
        end
    @testset "TrueSkill initalization" begin
        composition = [ [["aa"],["b"]], [["aa"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        priors = Dict{String,ttt.Player}()
        for k in ["aa", "b", "c"]
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 0.15*25.0/3) 
        end
        
        h = ttt.History(composition, results, [1,2,3], priors)
        
        @test isapprox(ttt.posterior(h.batches[1],"aa"),ttt.Gaussian(29.205,7.19448),1e-3)
        
        observed = h.batches[2].skills["aa"].forward.sigma 
        gamma = 0.15*25.0/3
        expected = sqrt((gamma*1)^2 + ttt.posterior(h.batches[1],"aa").sigma^2)
        @test isapprox(observed, expected)
        
        observed = ttt.posterior(h.batches[2],"aa")
        
        g = ttt.Game(ttt.within_priors(h.batches[2], 1),[0.,1.])
        expected = ttt.posteriors(g)[1][1]
        @test isapprox(observed, expected, 1e-7)
    end
    @testset "One-batch history" begin
        composition = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
        results = [[1.,0.],[1.,0.],[1.,0.]]
        times = [1,1,1]
        priors = Dict{String,ttt.Player}()
        for k in ["aj", "bj", "cj"]
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 0.15*25.0/3) 
        end
        
        h1 = ttt.History(composition,results, times, priors)
        # TrueSkill
        @test isapprox(ttt.posterior(h1.batches[1],"aj"),ttt.Gaussian(22.904,6.010),1e-3)
        @test isapprox(ttt.posterior(h1.batches[1],"cj"),ttt.Gaussian(25.110,5.866),1e-3)
        # TTT
        step , i = ttt.convergence(h1)
        @test isapprox(ttt.posterior(h1.batches[1],"aj"),ttt.Gaussian(25.000,5.419),1e-3)
        @test isapprox(ttt.posterior(h1.batches[1],"cj"),ttt.Gaussian(25.000,5.419),1e-3)
        
        priors = Dict{String,ttt.Player}()
        for k in ["aj", "bj", "cj"]
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300) 
        end
        h2 = ttt.History(composition,results, [1,2,3], priors  )
        # TrueSkill
        @test isapprox(ttt.posterior(h2.batches[3],"aj"),ttt.Gaussian(22.904,6.011),1e-3)
        @test isapprox(ttt.posterior(h2.batches[3],"cj"),ttt.Gaussian(25.111,5.867),1e-3)
        # TTT
        step , i = ttt.convergence(h2)
        @test isapprox(ttt.posterior(h2.batches[3],"aj"),ttt.Gaussian(24.999,5.420),1e-3)
        @test isapprox(ttt.posterior(h2.batches[3],"cj"),ttt.Gaussian(25.001,5.420),1e-3)
    end
    @testset "TrueSkill Through Time" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        priors = Dict{String,ttt.Player}()
        for k in ["a", "b", "c"]
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300) 
        end
        
        h = ttt.History(composition, results, Int64[], priors)
        step , iter = ttt.convergence(h)
        @test (h.batches[3].skills["b"].elapsed == 1) & (h.batches[3].skills["c"].elapsed == 1)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(25.0002673,5.41938162),1e-5)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(24.999465,5.419425831),1e-5)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian(25.00053219,5.419696790),1e-5)
        #ttt.learning_curves(h)
    end
    @testset "Env TTT" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        
        h = ttt.History(composition=composition, results=results, mu=25.,sigma=25.0/3, beta=25.0/6, gamma=25.0/300)
        step , iter = ttt.convergence(h)
        @test (h.batches[3].skills["b"].elapsed == 1) & (h.batches[3].skills["c"].elapsed == 1)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(25.0002673,5.41938162),1e-5)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(24.999465,5.419425831),1e-5)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian(25.00053219,5.419696790),1e-5)
        
    end
    @testset "Env 0 TTT" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        
        @time h = ttt.History(composition, results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        @time step , iter = ttt.convergence(h, iterations=100)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian( 0.001,2.395),1e-3)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(-0.001,2.396),1e-3)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian( 0.001,2.396),1e-3)
        
        composition = [ [["a"],["b"]], [["c"],["a"]] , [["b"],["c"]] ]
        @time h = ttt.History(composition=composition, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        @time step , iter = ttt.convergence(h, iterations=100)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian( 0.001,2.395),1e-3)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(-0.001,2.396),1e-3)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian( 0.001,2.396),1e-3)
        
        @time h = ttt.History(composition, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        @time step , iter = ttt.convergence(h, iterations=100)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian( 0.001,2.395),1e-3)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(-0.001,2.396),1e-3)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian( 0.001,2.396),1e-3)        
    end
    @testset "Env 0 TTT" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        
        @time h = ttt.History(composition=composition, results=results, times = [0, 10, 20], mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        @time step , iter = ttt.convergence(h, iterations=100)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian( 0.005,2.404),1e-3)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(-0.016,2.404),1e-3)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian( 0.017,2.406),1e-3)
    end
    @testset "Teams" begin
        composition = [ [["a","b"],["c","d"]], [["e","f"] , ["b","c"]], [["a","d"], ["e","f"]]  ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        
        h = ttt.History(composition=composition, results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        trueskill_log_evidence = ttt.log_evidence(h)
        trueskill_log_evidence_online =  ttt.log_evidence(h, online=true)
        @test isapprox(trueskill_log_evidence , trueskill_log_evidence_online ,rtol=1e-5)
        
        @test isapprox(ttt.posterior(h.batches[1],"b").mu, -1*ttt.posterior(h.batches[1],"c").mu,rtol=1e-5)
        
        evidence_second_event = exp(ttt.log_evidence(h, agents = ["b"]))*2
        @test isapprox(0.5,evidence_second_event,rtol=1e-5)
        evidence_third_event = exp(ttt.log_evidence(h, agents = ["a"]))*2
        @test isapprox(0.669885,evidence_third_event ,rtol=1e-5)
        
        step , iter = ttt.convergence(h)
        
        loocv_hat = exp(ttt.log_evidence(h))
        p_d_m_hat = exp(ttt.log_evidence(h, online=true))
        
        @test isapprox( ttt.posterior(h.batches[1],"a"), ttt.posterior(h.batches[1],"b"), 1e-3)
        @test isapprox( ttt.posterior(h.batches[1],"c"), ttt.posterior(h.batches[1],"d"), 1e-3)
        @test isapprox( ttt.posterior(h.batches[2],"e"), ttt.posterior(h.batches[2],"f"), 1e-3)
        
        @test isapprox( ttt.posterior(h.batches[1],"a"), ttt.Gaussian(mu=4.085 ,sigma=5.107), 1e-3 )
        @test isapprox( ttt.posterior(h.batches[1],"c"), ttt.Gaussian(mu=-0.533 ,sigma=5.107), 1e-3)
        @test isapprox( ttt.posterior(h.batches[3],"e"), ttt.Gaussian(mu=-3.552 ,sigma=5.155), 1e-3 )
    end
    @testset "sigma-beta 0" begin
        composition = [ [["a","a_b","b"],["c","c_d","d"]]
                 , [["e","e_f","f"],["b","b_c","c"]]
                 , [["a","a_d","d"],["e","e_f","f"]]  ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        priors = Dict{String,ttt.Player}()
        for k in ["a_b", "c_d", "e_f", "b_c", "a_d", "e_f"]
            priors[k] = ttt.Player(prior=ttt.Gaussian(mu=0.0, sigma=1e-7), beta=0.0, gamma=0.2) 
        end
        
        @time h = ttt.History(composition=composition, results=results, priors=priors, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        @time step , iter = ttt.convergence(h)
        @test isapprox( ttt.posterior(h.batches[1],"a_b"), ttt.Gaussian(mu=0.0 ,sigma=0.0), 1e-4 )
        @test isapprox( ttt.posterior(h.batches[3],"e_f"), ttt.Gaussian(mu=-0.002 ,sigma=0.2), 1e-4)
    end
    @testset "Memory Size" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        
        h = ttt.History(composition=composition, results=results, times = [0, 10, 20], mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        @test Base.summarysize(h) < 3850
        @test Base.summarysize(h.batches) - Base.summarysize(h.agents) < 3100
        @test Base.summarysize(h.agents) < 700
        @test Base.summarysize(h.batches[2]) - Base.summarysize(h.agents)  < 1000
        
        @test Base.summarysize(h) < Base.summarysize(composition)  * 7
        
        @test Base.summarysize(h.batches[1].skills) == 618
        @test Base.summarysize(h.batches[1].events) == 362
    end
    @testset "Learning curves" begin
        composition = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
        results = [[1.,0.],[1.,0.],[1.,0.]]    
        priors = Dict{String,ttt.Player}()
        for k in ["aj", "bj", "cj"]
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300) 
        end
        h = ttt.History(composition, results, [5,6,7], priors)
        ttt.convergence(h)
        lc = ttt.learning_curves(h)
        
        @test lc["aj"][1][1] == 5
        @test lc["aj"][end][1] == 7
        @test isapprox(lc["aj"][end][2],ttt.Gaussian(24.999,5.420),1e-3)
        @test isapprox(lc["cj"][end][2],ttt.Gaussian(25.001,5.420),13-3)
    end
    @testset "Add events into History 1" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        
        h = ttt.History(composition=composition, results=results, mu=0.0, sigma=2.0, beta=1.0, gamma=0.0)
        step , iter = ttt.convergence(h)
        @test (h.batches[3].skills["b"].elapsed == 1) & (h.batches[3].skills["c"].elapsed == 1)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(0.0,1.301),1e-3)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(0.0,1.301),1e-3)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian(0.0,1.301),1e-3)
        ttt.add_events(h, composition, results)
        
        @test length(h.batches) == 6
        @test [ttt.get_composition(b.events) for b in h.batches] == [ [[["a"], ["b"]]], [[["a"], ["c"]]], [[["b"], ["c"]]]
        , [[["a"], ["b"]]], [[["a"], ["c"]]], [[["b"], ["c"]]] ]

        
        step , iter = ttt.convergence(h)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(0.0,0.931),1e-3)
        @test isapprox(ttt.posterior(h.batches[4],"a"),ttt.Gaussian(0.0,0.931),1e-3)
        @test isapprox(ttt.posterior(h.batches[4],"b"),ttt.Gaussian(0.0,0.931),1e-3)
        @test isapprox(ttt.posterior(h.batches[6],"b"),ttt.Gaussian(0.0,0.931),1e-3)
        
        
        composition = [ [["a"],["b"]], [["c"],["a"]] , [["b"],["c"]] ]
        
        h = ttt.History(composition=composition, mu=0.0, sigma=2.0, beta=1.0, gamma=0.0)
        step , iter = ttt.convergence(h)
        @test (h.batches[3].skills["b"].elapsed == 1) & (h.batches[3].skills["c"].elapsed == 1)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(0.0,1.301),1e-3)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.Gaussian(0.0,1.301),1e-3)
        @test isapprox(ttt.posterior(h.batches[3],"b"),ttt.Gaussian(0.0,1.301),1e-3)
        ttt.add_events(h, composition)
        
        @test length(h.batches) == 6
        @test [ttt.get_composition(b.events) for b in h.batches] == [ [[["a"], ["b"]]], [[["c"], ["a"]]], [[["b"], ["c"]]]
        , [[["a"], ["b"]]], [[["c"], ["a"]]], [[["b"], ["c"]]] ]

        
        step , iter = ttt.convergence(h)
        @test isapprox(ttt.posterior(h.batches[1],"a"),ttt.Gaussian(0.0,0.931),1e-3)
        @test isapprox(ttt.posterior(h.batches[4],"a"),ttt.Gaussian(0.0,0.931),1e-3)
        @test isapprox(ttt.posterior(h.batches[4],"b"),ttt.Gaussian(0.0,0.931),1e-3)
        @test isapprox(ttt.posterior(h.batches[6],"b"),ttt.Gaussian(0.0,0.931),1e-3)
    end
    @testset "Add events into History 2" begin
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1.,0.],[0.,1.],[1.,0.]]
        times = [0,10,20]
        h = ttt.History(composition=composition, results=results, times=times, mu=0.0, sigma=2.0, beta=1.0, gamma=0.0)
        step , iter = ttt.convergence(h)
        times = [15,10,0]
        ttt.add_events(h, composition, results, times)
        @test length(h.batches) == 4
        @test [length(b) for b in h.batches] == [2,2,1,1]
        @test [ttt.get_composition(b.events) for b in h.batches] == [ 
        [[["a"], ["b"]], [["b"], ["c"]]] ,
        [[["a"], ["c"]], [["a"], ["c"]]] ,
        [[["a"], ["b"]]] ,
        [[["b"], ["c"]]]
        ]
        @test [ttt.get_results(b.events) for b in h.batches] == [ [[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]], [[1., 0.]], [[1., 0.]] ]
        @test (h.batches[1].skills["c"].elapsed == 0) & (h.batches[end].skills["c"].elapsed == 10)
        @test (h.batches[1].skills["a"].elapsed == 0) & (h.batches[3].skills["a"].elapsed == 5)
        @test (h.batches[1].skills["b"].elapsed == 0) & (h.batches[end].skills["b"].elapsed == 5)
        
        step , iter = ttt.convergence(h)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.posterior(h.batches[end],"b"),1e-4)
        @test isapprox(ttt.posterior(h.batches[1],"c"),ttt.posterior(h.batches[end],"c"),1e-4)
        @test isapprox(ttt.posterior(h.batches[1],"c"),ttt.posterior(h.batches[1],"b"),1e-4)
        
        # # # # # # # # #
        
        composition = [ [["a"],["b"]], [["c"],["a"]] , [["b"],["c"]] ]
        times = [0,10,20]
        h = ttt.History(composition=composition, times=times, mu=0.0, sigma=2.0, beta=1.0, gamma=0.0)
        step , iter = ttt.convergence(h)
        times = [15,10,0]
        ttt.add_events(h, composition, times=times)
        @test length(h.batches) == 4
        @test [length(b) for b in h.batches] == [2,2,1,1]
        @test [ttt.get_composition(b.events) for b in h.batches] == [ 
        [[["a"], ["b"]], [["b"], ["c"]]] ,
        [[["c"], ["a"]], [["c"], ["a"]]] ,
        [[["a"], ["b"]]] ,
        [[["b"], ["c"]]]
        ]
        @test [ttt.get_results(b.events) for b in h.batches] == [ [[1., 0.], [1., 0.]], [[1., 0.], [1., 0.]], [[1., 0.]], [[1., 0.]] ]
        @test (h.batches[1].skills["c"].elapsed == 0) & (h.batches[end].skills["c"].elapsed == 10)
        @test (h.batches[1].skills["a"].elapsed == 0) & (h.batches[3].skills["a"].elapsed == 5)
        @test (h.batches[1].skills["b"].elapsed == 0) & (h.batches[end].skills["b"].elapsed == 5)
        
        step , iter = ttt.convergence(h)
        @test isapprox(ttt.posterior(h.batches[1],"b"),ttt.posterior(h.batches[end],"b"),1e-4)
        @test isapprox(ttt.posterior(h.batches[1],"c"),ttt.posterior(h.batches[end],"c"),1e-4)
        @test isapprox(ttt.posterior(h.batches[1],"c"),ttt.posterior(h.batches[1],"b"),1e-4)
        
    end
    @testset "Add events into History 2" begin
        composition = [[["a"],["b"]],[["b"],["a"]]]
        h = ttt.History(composition)
        p_d_m_2 = exp(ttt.log_evidence(h))*2
        @test isapprox(p_d_m_2, 0.17650911, rtol=1e-5)
        @test isapprox(p_d_m_2, exp(ttt.log_evidence(h, online=true))*2, rtol=1e-5)
        @test isapprox(p_d_m_2, exp(ttt.log_evidence(h, online=true, agents = ["a"]))*2, rtol=1e-5)
        @test isapprox(p_d_m_2, exp(ttt.log_evidence(h, online=false, agents = ["a"]))*2, rtol=1e-5)
        
        ttt.convergence(h,iterations=11)
        loocv_approx_2 = sqrt(exp(ttt.log_evidence(h)))
        @test isapprox(loocv_approx_2, 0.001976774, rtol=1e-5)
        
        p_d_m_approx_2 = exp(ttt.log_evidence(h, online=true))*2
        @test loocv_approx_2-p_d_m_approx_2 < 1e-4
        @test isapprox(loocv_approx_2, exp(ttt.log_evidence(h, online=true, agents= ["b"]))*2, atol=1e-5)
        
        
    end
    @testset "1vs1 with weights" begin
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wa = [1.0]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wb = [2.0]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(30.625173, 7.765472), 1e-4)
        @test isapprox(post[2][1], ttt.Gaussian(13.749653, 5.733840), 1e-4)

        wa = [1.0]
        wb = [0.7]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(27.630080, 7.206676), 1e-4)
        @test isapprox(post[2][1], ttt.Gaussian(23.158943, 7.801628), 1e-4)

        wa = [1.6]
        wb = [0.7]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(26.142438, 7.573088), 1e-4)
        @test isapprox(post[2][1], ttt.Gaussian(24.500183, 8.193278), 1e-4)
        
        wa = [1.0]; wb = [0.0]
        ta = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        tb = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(5.557176746, 4.0527906913), 1e-3)
        @test isapprox(post[2][1], ttt.Gaussian(2.0, 6.0), 1e-4)
        # NOTA: trueskill original tiene probelmas en la aproximación: post[2][1].mu = 1.999644 
        
        wa = [1.0]; wb = [-1.0]
        ta = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        tb = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], post[2][1], 1e-4)
        end
    @testset "NvsN with weights" begin
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0), ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wa = [0.4, 0.8]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0), ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wb = [0.9, 0.6]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(27.539023, 8.129639), 1e-4)
        @test isapprox(post[1][2], ttt.Gaussian(30.078046, 7.485372), 1e-4)
        @test isapprox(post[2][1], ttt.Gaussian(19.287197, 7.243465), 1e-4)
        @test isapprox(post[2][2], ttt.Gaussian(21.191465, 7.867608), 1e-4)

        wa = [1.3, 1.5]
        wb = [0.7, 0.4]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(25.190190, 8.220511), 1e-4)
        @test isapprox(post[1][2], ttt.Gaussian(25.219450, 8.182783), 1e-4)
        @test isapprox(post[2][1], ttt.Gaussian(24.897589, 8.300779), 1e-4)
        @test isapprox(post[2][2], ttt.Gaussian(24.941479, 8.322717), 1e-4)

        wa = [1.6, 0.2]
        wb = [0.7, 2.4]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(31.674697, 7.501180), 1e-4)
        @test isapprox(post[1][2], ttt.Gaussian(25.834337, 8.320970), 1e-4)
        @test isapprox(post[2][1], ttt.Gaussian(22.079819, 8.180607), 1e-4)
        @test isapprox(post[2][2], ttt.Gaussian(14.987953, 6.308469), 1e-4)

    
        tc = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        g = ttt.Game([ta,tc])
        post_2vs1 = ttt.posteriors(g)
        
        wa = [1.0, 1.0]
        wb = [1.0, 0.0]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], post_2vs1[1][1], 1e-4)
        @test isapprox(post[1][2], post_2vs1[1][2], 1e-4)
        @test isapprox(post[2][1], post_2vs1[2][1], 1e-4)
        @test isapprox(post[2][2], tb[2].prior, 1e-4)
    end
    @testset "1vs1 TTT with weights" begin
        composition = [[["a"],["b"]], [["b"],["a"]]]
        weights = [[[5.0],[4.0]],[[5.0],[4.0]]]
        h = ttt.History(composition, mu=2.0, beta=1.0, sigma=6.0, gamma=0.0, weights=weights)
        lc = ttt.learning_curves(h)
        @test isapprox(lc["a"][1][2], ttt.Gaussian(5.53765944, 4.758722), 1e-4)
        @test isapprox(lc["b"][1][2], ttt.Gaussian(-0.83012755, 5.2395689), 1e-4)
        @test isapprox(lc["a"][2][2], ttt.Gaussian(1.7922776, 4.099566689), 1e-4)
        @test isapprox(lc["b"][2][2], ttt.Gaussian(4.8455331752, 3.7476161), 1e-4)
        
        ttt.convergence(h)
        lc = ttt.learning_curves(h)
        @test isapprox(lc["a"][1][2], ttt.Gaussian(lc["a"][1][2].mu, lc["a"][1][2].sigma), 1e-4)
        @test isapprox(lc["b"][1][2], ttt.Gaussian(lc["a"][1][2].mu, lc["a"][1][2].sigma), 1e-4)
        @test isapprox(lc["a"][2][2], ttt.Gaussian(lc["a"][1][2].mu, lc["a"][1][2].sigma), 1e-4)
        @test isapprox(lc["b"][2][2], ttt.Gaussian(lc["a"][1][2].mu, lc["a"][1][2].sigma), 1e-4)
        
        composition = [[["a"],["b"]], [["b"],["a"]]]
        weights = [[[1.0],[4.0]],[[5.0],[4.0]]]
        h = ttt.History(composition, mu=2.0, beta=1.0, sigma=6.0, gamma=0.0, weights=weights)
        lc = ttt.learning_curves(h)
        
    end
end


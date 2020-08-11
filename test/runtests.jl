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
    @testset "1vs1" begin
        ta = [ttt.Rating()]
        tb = [ttt.Rating()]
        g = ttt.Game([ta,tb], [1,0])
        post = ttt.posteriors(g)
        @test isapprox(post[1][1], ttt.Gaussian(20.794779,7.194481), 1e-4) 
        @test isapprox(post[2][1], ttt.Gaussian(29.205220,7.194481), 1e-4)
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
        post[1][1].sigma
        post[1][2].sigma
        post[2][1].sigma
        @test isapprox(post[1][1],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[1][2],ttt.Gaussian(15.000,0.9916),1e-3)
        @test isapprox(post[2][1],ttt.Gaussian(30.000,1.9320),1e-3)
    end
    
end


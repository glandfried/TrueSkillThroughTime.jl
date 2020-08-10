#include('../src')
include("./TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill
using Test

begin
    @test isapprox(ttt.ppf(ttt.N01,0.3),-0.52440044)
end
begin
     a = isapprox(ttt.compute_margin(0.25,2),1.8776005988)
     b = isapprox(ttt.compute_margin(0.25,3),2.29958170)
     c = isapprox(ttt.compute_margin(0.0,3),2.7134875810435737e-07)
     d = isapprox(ttt.compute_margin(1.0,3),Inf)
     @test a & b & c & d
end
begin
    a = ttt.trunc(ttt.N01,0.0,false)
    b = ttt.Gaussian(0.7978845368663289,0.6028103066716792)
    @test isapprox(a,b,1e-4) 
end
begin
    a = ttt.Gaussian(0.,sqrt(2)*ttt.BETA)
    margin = 1.8776005988
    res = ttt.trunc(a,margin,true)
    @test isapprox(res, ttt.Gaussian(0.,1.076707),1e-5)
end
begin
    ta = [ttt.Rating()]
    tb = [ttt.Rating()]
    g = ttt.Game([ta,tb], [1,0])
    post = ttt.posteriors(g,0.)
    a = ttt.Gaussian(20.79477925612302,7.194481422570443)
    b = ttt.Gaussian(29.20522074387697,7.194481422570443)
    testa = isapprox(post[1][1], a, 1e-4) 
    testb = isapprox(post[2][1], b, 1e-4)
    @test testa & testb
end
begin
    g = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2])
    post = ttt.posteriors(g,0.)
    testa = isapprox(post[1][1],ttt.Gaussian(25.000000,6.238469796),1e-5)
    testb = isapprox(post[2][1],ttt.Gaussian(31.3113582213,6.69881865) ,1e-5)
    @test testa & testb
end
begin
    ta = [ttt.Rating()]
    tb = [ttt.Rating()]
    g = ttt.Game([ta,tb], [0,0])
    post = ttt.posteriors(g,0.25)
    post[1][1].sigma
end



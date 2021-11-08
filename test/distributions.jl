using TrueSkillThroughTime
ttt = TrueSkillThroughTime
using Distributions
using Test

function approximate_truncated(N::Normal, margin::Float64, tie::Bool)
    if !tie
        tN = TruncatedNormal(mean(N), std(N), margin, Inf)
    else
        tN = TruncatedNormal(mean(N), std(N), -margin, margin)
    end
    Normal(mean(tN), std(tN))
end

N = Normal()
G = ttt.Gaussian(0.0,1.0)
ttt.approx(G,0.0,false)
ttt.approx(G,1.0,true)
approximate_truncated(N,0.0,false)
approximate_truncated(N,1.0,true)

N = Normal(3.0, 2.0)
G = ttt.Gaussian(3.0,2.0)
@timev ttt.approx(G,0.0,false)
@timev approximate_truncated(N,0.0,false)

N = Normal(2.0, 4.0)
G = ttt.Gaussian(2.0,4.0)
@timev ttt.approx(G,1.0,true)
@timev approximate_truncated(N,1.0,true)


include("../src/TrueSkill.jl")
using .TrueSkill
global const ttt = TrueSkill

# Code 1
mu = 0.0; sigma = 6.0; beta = 1.0; gamma = 0.0
p1 = ttt.Player(ttt.Gaussian(mu, sigma), beta, gamma); p2 = ttt.Player(ttt.Gaussian(mu, sigma), beta, gamma)
p3 = ttt.Player(ttt.Gaussian(mu, sigma), beta, gamma); p4 = ttt.Player(ttt.Gaussian(mu, sigma), beta, gamma)

# Code 2
team_a = [ p1, p2 ]
team_b = [ p3, p4 ]
teams = [team_a, team_b]
g = ttt.Game(teams)
#g = ttt.Game(teams,[0.,0.])

# Code 3
lhs = g.likelihoods
ev = g.evidence
ev = round(ev, digits=3)
print(ev)

# Code 4
pos = ttt.posteriors(g)
print(pos[1][1])
print(lhs[1][1] * p1.prior)

# Code 5
ta = [p1]
tb = [p2, p3]
tc = [p4]
teams = [ta, tb, tc]
result = [1., 0., 0.]
g = ttt.Game(teams, result, p_draw=0.25)

# Code 6
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = ttt.History(composition)

# Code 7
lc = ttt.learning_curves(h)
print(lc["a"])
print(lc["b"])

# Code 8
ttt.convergence(h)
lc = ttt.learning_curves(h)
print(lc["a"])
print(lc["b"])




############

using Plots
using Random
Random.seed!(999)
function skill(experiencia::Int64, alpha::Float64=0.0075, media::Int64 = 500)
    return 1/(1+exp(alpha*(-experiencia+media))) *2
end

mean_agent = [skill(i) for i in 1:1000] 
mean_target = Random.randn(1000)*0.5 .+ mean_agent 
perf_target = Random.randn(1000) .+ mean_target
perf_agent = Random.randn(1000) .+ mean_agent


composition = [[["a"], [string(i)]] for i in 1:1000]
results = [ perf_agent[i] > perf_target[i] ? [1.,0.] : [0.,1.] for i in 1:1000 ]
times = [i for i in 1:1000 ]
priors = Dict{String,ttt.Player}()
for k in 1:1000
    priors[string(k)] = ttt.Player(ttt.Gaussian(mean_target[k], 0.1)) 
end
# gammas = [gamma for gamma in 0.001:0.001:0.04]
# evidencias = []
# for g in gammas 
#     h = ttt.History(composition, results, times, priors, gamma=g)
#     ttt.convergence(h)
#     push!(evidencias, ttt.log_evidence(h))
# end
# 0.017==gammas[argmax(evidencias)]
h = ttt.History(composition, results, times, priors, sigma=6.0, gamma=0.017)
ttt.convergence(h)
    
mu = [tp[2].mu for tp in ttt.learning_curves(h)["a"]]
sigma = [tp[2].sigma for tp in ttt.learning_curves(h)["a"]]

plot(mu)
plot!(mean_agent)

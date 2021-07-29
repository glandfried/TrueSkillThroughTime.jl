include("../src/TrueSkillThroughTime.jl")
using .TrueSkillThroughTime
global const ttt = TrueSkillThroughTime
using CSV; using Dates

println("Code 1")
mu = 0.0; sigma = 6.0; beta = 1.0; gamma = 0.03; draw = 0.0

println("Code 2")
a1 = ttt.Player(ttt.Gaussian(mu, sigma), beta, gamma); a2 = ttt.Player()
a3 = ttt.Player(); a4 = ttt.Player()

println("Code 3")
team_a = [ a1, a2 ]
team_b = [ a3, a4 ]
teams = [team_a, team_b]
g = ttt.Game(teams)

@time ttt.posteriors(ttt.Game(teams))

println("Code 4")
lhs = g.likelihoods
ev = g.evidence
ev = round(ev, digits=3)
println(ev)

println("Code 5")
pos = ttt.posteriors(g)
print(pos[1][1])
print(lhs[1][1] * a1.prior)

println("Code 6")
ta = [a1]
tb = [a2, a3]
tc = [a4]
teams_3 = [ta, tb, tc]
result = [1., 0., 0.]
g = ttt.Game(teams_3, result, p_draw=0.25)

@time ttt.posteriors(ttt.Game(teams_3, result, p_draw=0.25))

println("Code 7")
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = ttt.History(composition, gamma = 0.0)

@time ttt.History(composition)

println("Code 8")
lc = ttt.learning_curves(h)
print(lc["a"])
print(lc["b"])

println("Code 9")
ttt.convergence(h)
lc = ttt.learning_curves(h)
print(lc["a"])
print(lc["b"])

@time ttt.convergence(h, iterations=1)

println("Code 10")
using Random; Random.seed!(999); N = 1000
function skill(experience, middle, maximum, slope)
    return maximum/(1+exp(slope*(-experience+middle)))
end
target = skill.(1:N, 500, 2, 0.0075)
opponents = Random.randn.(1000)*0.5 .+ target

println("Code 11")
composition = [[["a"], [string(i)]] for i in 1:N]
results = [ r ? [1.,0.] : [0.,1.] for r in (Random.randn(N).+target .> Random.randn(N).+opponents) ]
times = [i for i in 1:N]
priors = Dict{String,ttt.Player}()
for i in 1:N  priors[string(i)] = ttt.Player(ttt.Gaussian(opponents[i], 0.2))  end
@time h = ttt.History(composition, results, times, priors, gamma=0.015)
ttt.convergence(h)
@time ttt.convergence(h, iterations=1)

mu = [tp[2].mu for tp in ttt.learning_curves(h)["a"]]

println("Code 12")
data = CSV.read("input/history.csv")

times = Dates.value.(data[:,"time_start"] .- Date("1900-1-1"))
composition = [ r.double == "t" ? [[r.w1_id,r.w2_id],[r.l1_id,r.l2_id]] :
[[r.w1_id],[r.l1_id]] for r in eachrow(data) ]

h = ttt.History(composition=composition, times = times, sigma = 1.6, gamma = 0.036)
ttt.convergence(h,epsilon=0.01, iterations=10)

println("Code 13")
players = Set(vcat((composition...)...))
priors = Dict([(p, ttt.Player(ttt.Gaussian(0., 1.6), 1.0, 0.036) ) for p in players])

composition_ground = [ r.double == "t" ? [[r.w1_id, r.w1_id*r.ground, r.w2_id, r.w2_id*r.ground],[r.l1_id, r.l1_id*r.ground, r.l2_id, r.l2_id*r.ground]] : [[r.w1_id, r.w1_id*r.ground],[r.l1_id, r.l1_id*r.ground]] for r in eachrow(data) ]

h_ground = ttt.History(composition=composition_ground, times = times, sigma = 1.0, gamma = 0.01, beta = 0.0, priors = priors)
ttt.convergence(h_ground,epsilon=0.01, iterations=10)

println("Code 14")
N1 = ttt.Gaussian(mu = 1.0, sigma = 1.0); N2 = ttt.Gaussian(1.0, 2.0)

println("Code 15")
p1 = ttt.performance(a1)
p2 = ttt.performance(a2)
p3 = ttt.performance(a3)
p4 = ttt.performance(a4)

println("Code 16")
ta = p1 + p2; tb = p3 + p4

println("Code 17")
d = ta - tb

println("Code 18")
e = 1.0 - ttt.cdf(d, 0.0)

println("Code 19")
na = length(team_a)
nb = length(team_b)
sd = sqrt(na + nb)*beta
p_draw = 0.25
margin = ttt.compute_margin(p_draw, sd)

println("Code 20")
g = ttt.Game(teams, p_draw = 0.25)
post = ttt.posteriors(g)

println("Code 21")
d_approx = ttt.approx(d, margin, false)

println("Code 22")
approx_lh_d = d_approx / d

println("Code 23")
mu = a1.prior.mu
sigma2 = a1.prior.sigma^2
phi = d.mu
v2 = d.sigma^2
phi_div = approx_lh_d.mu
v2_div = approx_lh_d.sigma^2
prior = a1.prior
posterior = post[1][1]
println( prior * ttt.Gaussian(mu-phi+phi_div, sqrt(v2 + v2_div - sigma2)) )

println(posterior)


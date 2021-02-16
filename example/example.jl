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

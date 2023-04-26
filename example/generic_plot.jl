using TrueSkillThroughTime
global const ttt = TrueSkillThroughTime
using Plots

# Solve you own example
composition = [[[string(rand('a':'e'))], [string(rand('a':'e'))]] for i in 1:1000]
h = ttt.History(composition=composition, gamma=0.03, sigma=1.0)
ttt.convergence(h)

# Plot all the learning_curves
lc = ttt.learning_curves(h)
pp = plot(xlabel="t", ylabel="mu", title="Learning Curves")
for (i, agent) in enumerate(keys(h.agents))#agent="a"#i=1
    t = [v[1] for v in lc[agent] ]
    mu = [v[2].mu for v in lc[agent] ]
    sigma = [v[2].sigma for v in lc[agent] ]
    plot!(t, mu, color=i, label=agent)
    plot!(t, mu.+sigma, fillrange=mu.-sigma, alpha=0.2,color=i, label=false)
end
display(pp)


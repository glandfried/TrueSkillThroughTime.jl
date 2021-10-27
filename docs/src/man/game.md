# The `Game` class

```@setup all
using TrueSkillThroughTime
ttt = TrueSkillThroughTime
```

We use the `Game` class to model events and perform inference.

```@contents
Pages = ["game.md"]
```

```@docs
Game
```

Let us return to the example seen on the first page of this manual.

```@example all
a1 = ttt.Player(); a2 = ttt.Player(); a3 = ttt.Player(); a4 = ttt.Player()
team_a = [ a1, a2 ]
team_b = [ a3, a4 ]
g = ttt.Game([team_a, team_b])
g.teams
```
where the teams' order in the list implicitly defines the game's result: the teams appearing first in the list (lower index) beat those appearing later (higher index). 

## Evidence and likelihood

During the initialization, the `Game` class computes the prior prediction of the observed result (the `evidence` property) and the approximate likelihood of each player (the `likelihoods` property).

```@example all
lhs = g.likelihoods
round(g.evidence, digits=3)
```

In this case, the evidence is $0.5$ because both teams had the same prior skill estimates.

## Posterior 

The method `posteriors()` of class `Game` to compute the posteriors.

```@docs
posteriors
```

```@example all
pos = ttt.posteriors(g)
pos[1][1]
```

Posteriors can also be found by manually multiplying the likelihoods and priors. 

```@example all
lhs[1][1] * a1.prior
```

## Team performance

```@docs
performance(G::Game,i::Int64)
```

We can obtain the expected performance of the first team. 

```@example all
ttt.performance(g,1)
```

## Full example

We now analyze a more complex example in which the same four players participate in a multi-team game.
The players are organized into three teams of different sizes: two teams with only one player and the other with two players. 
The result has a single winning team and a tie between the other two losing teams.
Unlike the previous example, we need to use a draw probability greater than zero.

```@example all
ta = [a1]
tb = [a2, a3]
tc = [a4]
teams_3 = [ta, tb, tc]
result = [1., 0., 0.]
g = ttt.Game(teams_3, result, p_draw=0.25)
g.result
```

The team with the highest score is the winner, and the teams with the same score are tied.
In this way, we can specify any outcome including global draws.
The evidence and posteriors can be queried in the same way as before.

```@example all
ttt.posteriors(g)
```

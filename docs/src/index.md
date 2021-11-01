# TrueSkillThroughTime.jl

**The state-of-the-art skill model**: _Individual learning curves with reliable initial estimates and guaranteed comparability between distant estimates._

## Install

    using Pkg
    Pkg.add("TrueSkillThroughTime")

```@setup all
using TrueSkillThroughTime
ttt = TrueSkillThroughTime
```

The we can use it.
```
using TrueSkillThroughTime
ttt = TrueSkillThroughTime
```

- To appreciate the advantages of TrueSkill Through Time, scroll down to subsection [Ilustration](@ref ilustration)
- To quickly see how to use the package, scroll down to subsection [First examples](@ref first_examples).

## Index


```@contents
Pages = ["man/causal.md", "man/gaussian.md", "man/player.md", "man/game.md", "man/history.md", "man/examples.md"]
Depth = 1
```

## [Ilustration](@id ilustration)

To appreciate the advantages of TrueSkill Through Time, let's see how it works in a real case.
The following figure presents the estimated learning curves of some famous male players in ATP's history, which we identified using different colors (to see the source code go to section [Real examples](@ref real_example)).

![](assets/atp.png)

The top bar indicates which player was at the top of the ATP's ranking (the bar has no color when player number 1 is not included among the 10 players identified with colors).
There is a relative coincidence between the skill estimates and who is at any given moment at the top of the ATP rankings.
However, TrueSkill Through Time allows comparing the relative ability of players over time: the 10th player in the historical ATP's ranking, Hewitt, is a product of the window of opportunity that was opened in the year 2000; and the 4th most skilled player, Murray, is ranked 14th just above Nastase.

The **models commonly used in industry and academia** (TrueSkill, Glicko, Item-Response Theory) propagates information from past events to future events.
Because this approach is an ad-hoc procedure that does not arise from any probabilistic model, its estimates have a number of problems.

![](assets/atp_trueskill.png)

The advantage of TrueSkill Through Time lies in its temporal causal model, that links all historical activities in the same Bayesian network, guaranteeing reliable initial estimates and comparability between distant estimates.

## [First examples](@id first_examples)

We can update our skill estimates after a single event, or we can estimate the learning curves of all players from a history of events.
Let's see both cases.

### A single game

We use the `Game` class to model events and perform inference.
The features of the agents are defined within `Player` class.

```@example all
a1 = ttt.Player(); a2 = ttt.Player(); a3 = ttt.Player(); a4 = ttt.Player()
team_a = [ a1, a2 ]
team_b = [ a3, a4 ]
g = ttt.Game([team_a, team_b])
ttt.posteriors(g)
```

where the teams' order in the list implicitly defines the game's result: the teams appearing first in the list (lower index) beat those appearing later (higher index). 
This is one of the simplest usage examples.
Later on, we will learn how to explicitly specify the result, and others features.

### A history of events

We use the `History` class to compute the learning curves and predictions of a sequence of events.
We will define the composition of each game using the names of the agents (i.e. their identifiers).
In the following example, all agents (`"a", "b", "c"`) win one game and lose the other. 
The results will be implicitly defined by the order in which the game compositions are initialized: the teams appearing firstly in the list defeat those appearing later. 
By initializing `gamma = 0.0` we specify that skills do not change over time.
In this example, where all agents beat each other and their skills do not change over time, the data suggest that all agents have the same skill.

```@example all
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = ttt.History(composition, gamma=0.0)
```

After initialization, the `History` class immediately instantiates a new player for each name and activates the computation of the TrueSkill estimates (not yet TrueSkill Through Time).
To access them we can call the method `learning\_curves()`, which returns a dictionary indexed by the names of the agents.

```@example all
ttt.learning_curves(h)["a"]
```
```@example all
ttt.learning_curves(h)["b"]
```

Individual learning curves are lists of tuples: each tuple has the time of the estimate as the first component and the estimate itself as the second one.
Although in this example no player is stronger than the others, the TrueSkill estimates present strong variations between players.
TrueSkill Through Time solves TrueSkill's inability to obtain correct estimates by allowing the information to propagate throughout the system.
To compute them, we call the method `convergence()` of the `History` class.

```@example all
ttt.convergence(h)
ttt.learning_curves(h)["a"]
```
```@example all
ttt.learning_curves(h)["b"]
```

TrueSkill Through Time not only returns correct estimates (same for all players), they also have less uncertainty.

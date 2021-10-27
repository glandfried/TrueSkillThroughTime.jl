# The `Player` class

```@setup all
using TrueSkillThroughTime
ttt = TrueSkillThroughTime
```


```@contents
Pages = ["player.md"]
```

The features of the agents are defined within class `Player`: the prior Gaussian distribution characterized by the mean (`mu`) and the standard deviation (`sigma`), the standard deviation of the performance (`beta`), and the dynamic uncertainty of the skill (`gamma`). 

```@docs
Player
```

The default value of [`MU`](@ref), [`SIGMA`](@ref), [`BETA`](@ref) and [`GAMMA`](@ref) are 

```@repl all
a1 = ttt.Player()
```



```@repl all
a2 = ttt.Player(ttt.Gaussian(0.0, 1.0))
```

We can also create special players who have non-random performances (`beta=0.0`), and whose skills do not change over time (`gamma=0.0`).

```@repl all
a3 = ttt.Player(beta=0.0, gamma=0.0)
a3.beta
a3.gamma
```

## Performance

The performances $p$ are random variables around their unknown true skill $s$,

``p \sim \mathcal{N}(s,\beta^2)``

```@docs
performance(R::Player)
```

```@repl all
ttt.performance(a2)
ttt.performance(a3)
```


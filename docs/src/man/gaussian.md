#  [The `Gaussian` class](@id gaussian)

```@setup all
using TrueSkillThroughTime
ttt = TrueSkillThroughTime
```

The `Gaussian` class does most of the computation of the packages.


```@contents
Pages = ["gaussian.md"]
```

```@docs
Gaussian
```

The default value of [`MU`](@ref) and [`SIGMA`](@ref) are


```@example all 
N06 = ttt.Gaussian()
```

Others ways to create `Gaussian` objects

```@example all
N01 = ttt.Gaussian(sigma = 1.0)
N12 = ttt.Gaussian(1.0, 2.0)
Ninf = ttt.Gaussian(1.0,Inf)
println("mu: ", N01.mu, ", sigma: ", N01.sigma)
```


The class overwrites the addition `+`, subtraction `-`, product `*`, and division `/` to compute the marginal distributions used in the TrueSkill Through Time model.

## Product `*`

- ``k \mathcal{N}(x|\mu,\sigma^2) = \mathcal{N}(x|k\mu,(k\sigma)^2)``
- ``\mathcal{N}(x|\mu_1,\sigma_1^2)\mathcal{N}(x|\mu_2,\sigma_2^2) \propto \mathcal{N}(x|\mu_{*},\sigma_{*}^2)``

with $\frac{\mu_{*}}{\sigma_{*}^2} = \frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}$ and $\sigma_{*}^2 = (\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2})^{-1}$.

```@docs
*
```

```@repl all 
N06 * N12
N12 * 5.0
N12 * Ninf
```

## Division `/`

- ``\mathcal{N}(x|\mu_1,\sigma_1^2)/\mathcal{N}(x|\mu_2,\sigma_2^2) \propto \mathcal{N}(x|\mu_{\div},\sigma_{\div}^2)``

with $\frac{\mu_{\div}}{\sigma_{\div}^2} = \frac{\mu_1}{\sigma_1^2} - \frac{\mu_2}{\sigma_2^2}$ and $\sigma_{\div}^2 = (\frac{1}{\sigma_1^2} - \frac{1}{\sigma_2^2})^{-1}$.

```@docs
/
```

```@repl all 
N12 / N06
N12 / Ninf
```

## Addition `+`

- ``\iint \delta(t=x + y) \mathcal{N}(x|\mu_1, \sigma_1^2)\mathcal{N}(y|\mu_2, \sigma_2^2) dxdy =  \mathcal{N}(t|\mu_1+\mu_2,\sigma_1^2 + \sigma_2^2)``


```@docs
+
```

```@repl all 
N06 + N12
```

## Substraction `-`

- ``\iint \delta(t=x - y) \mathcal{N}(x|\mu_1, \sigma_1^2)\mathcal{N}(y|\mu_2, \sigma_2^2) dxdy =  \mathcal{N}(t|\mu_1-\mu_2,\sigma_1^2 + \sigma_2^2)`` 
```@docs
-
```

```@repl all 
N06 - N12
```

## Others methods

### isapprox

```@docs
isapprox
```

```@repl all
N06-N12 == ttt.Gaussian(mu=-1.0, sigma=6.324555)
ttt.isapprox(N06-N12, ttt.Gaussian(mu=-1.0, sigma=6.324555), 1e-6)
```

### forget

```@docs
forget
```

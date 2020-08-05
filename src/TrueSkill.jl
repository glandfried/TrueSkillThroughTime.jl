module TrueSkill

global const MU = 25.0::Float64
global const SIGMA = (MU/3)::Float64
global const BETA = (SIGMA / 2)::Float64
global const TAU = (SIGMA / 100)::Float64
global const DRAW_PROBABILITY = 0.0::Float64
global const EPSILON = 0.1::Float64
Base.@kwdef struct Gaussian
    mu::Float64 = 0
    sigma::Float64 = Inf
    #TODO: Domain() = if sigma < 0 DomainError() 
end
global const N01 = Gaussian(mu=0,sigma=1)
#
function pi(N::Gaussian)
    if N.sigma>0
        return N.sigma^-2
    else
        return Inf
    end
end
function tau(N::Gaussian)
    if N.sigma>0
        return pi(N)*N.mu
    else
        return Inf
    end
end
Base.show(io::IO, g::Gaussian) = print("Gaussian(mu=", round(g.mu,digits=3)," ,sigma=", round(g.sigma,digits=3), ")")
function erfc(x::Float64)
    #"""Complementary error function (thanks to http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = begin
        a = -0.82215223 + t * 0.17087277 
        b =  1.48851587 + t * a
        c = -1.13520398 + t * b
        d =  0.27886807 + t * c
        e = -0.18628806 + t * d
        f =  0.09678418 + t * e
        g =  0.37409196 + t * f
        h =  1.00002368 + t * g
        -z * z - 1.26551223 + t
        end
    if x < 0 r = 2. - r end
    return r
end
function cdf(N::Gaussian, x::Float64)
    z = -(x - N.mu) / (N.sigma * sqrt(2))
    return (0.5 * erfc(z))::Float64
end
function pdf(N::Gaussian, x::Float64)
    normalizer = (sqrt(2 * pi ) * N.sigma)^-1
    functional = exp(-(((x - N.mu)^2) / ((N.sigma ^2) * 2))) 
    return (normalizer * functional)::Float64
end
function trunc(N::Gaussian, margin::Float64, tie::Bool)
    #TODO: tie
    #draw_margin = calc_draw_margin(draw_probability, size, self)
    _alpha = (-margin-N.mu)/N.sigma
    _beta = (margin-N.mu)/N.sigma
    if !tie
        t = -_alpha
        v = pdf(N01,t) / cdf(N01,t)
        w = v * (v + t)
    else
        v = (pdf(N01,_alpha)-pdf(N01,_beta))/(cdf(N01,_beta)-cdf(N01,_alpha))
        u = (_alpha*pdf(N01,_alpha)-_beta*pdf(N01,_beta))/(cdf(N01,_beta)-cdf(N01,_alpha))
        w =  - ( u - v^2 ) 
    mu = N.mu + N.sigma * v
    sigma = N.sigma*sqrt(1-w)
    return Gaussian(mu=mu,sigma=sigma)
end
function Base.:+(N::Gaussian, M::Gaussian)
    mu = N.mu + M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu=mu, sigma=sigma )
end
function Base.:-(N::Gaussian, M::Gaussian)
    mu = N.mu - M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu=mu, sigma=sigma )
end
function Base.:*(N::Gaussian, M::Gaussian)
    _pi = pi(N) + pi(M)
    _tau = tau(N) + tau(M)
    return Gaussian(mu=_tau/_pi, sigma=sqrt(1/_pi))        
end
function Base.:/(N::Gaussian, M::Gaussian)
    _pi = pi(N) - pi(M)
    _tau = tau(N) - tau(M)
    return Gaussian(mu=_tau/_pi, sigma=sqrt(1/_pi))        
end
#
g1 = Gaussian(sigma=SIGMA)
g2 = Gaussian(mu=2.0,sigma=3.0)
@time g2/g1





end # module

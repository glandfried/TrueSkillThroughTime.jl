module TrueSkill

#import SpecialFunctions

global const MU = 25.0::Float64
global const SIGMA = (MU/3)::Float64
global const BETA = (SIGMA / 2)::Float64
global const TAU = (SIGMA / 100)::Float64
global const DRAW_PROBABILITY = 0.0::Float64
global const EPSILON = 0.1::Float64
global const sqrt2 = sqrt(2)
global const sqrt2pi = sqrt(2*pi)

function erfc(x::Float64)
    #"""Complementary error function (thanks to http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    r = begin
        a = -0.82215223 + t * 0.17087277 
        b =  1.48851587 + t * a
        c = -1.13520398 + t * b
        d =  0.27886807 + t * c
        e = -0.18628806 + t * d
        f =  0.09678418 + t * e
        g =  0.37409196 + t * f
        h =  1.00002368 + t * g
        t * exp(-z * z - 1.26551223 + t * h)
        end
    if x < 0
        r = 2.0 - r
    end
    return r
end
function erfcinv(y::Float64)
    """The inverse function of erfc."""
    if y >= 2
        return -Inf
    elseif y < 0
        throw(DomainError(y, "argument must be nonnegative"))
    elseif y == 0
        return Inf
    end
    zero_point = y < 1
    if ! zero_point 
        y = 2 - y
    end
    t = sqrt(-2 * log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in 0:2
        err = erfc(x) - y
        x += err / (1.12837916709551257 * exp(-(x^2)) - x * err)
    end
    if zero_point
        r = x
    else 
        r = -x
    end
    return r
end
struct Gaussian
    # TODO: support Gaussian(mu=0.0,sigma=1.0)
    mu::Float64
    sigma::Float64
    pi::Float64
    tau::Float64
    function Gaussian(mu::Float64, sigma::Float64)
        if sigma < 0
            error("sigma should be greater than 0")
        elseif sigma>0
            _pi = sigma^-2
            _tau = _pi * mu
        else
            _pi = Inf
            _tau = Inf
        end
        return new(mu, sigma, _pi, _tau)
    end
end
global const N01 = Gaussian(0.0, 1.0)
global const Ninf = Gaussian(0.0, Inf)

Base.show(io::IO, g::Gaussian) = print("Gaussian(mu=", round(g.mu,digits=3)," ,sigma=", round(g.sigma,digits=3), ")")
function cdf(N::Gaussian, x::Float64)
    z = -(x - N.mu) / (N.sigma * sqrt2)
    return (0.5 * erfc(z))::Float64
end
function pdf(N::Gaussian, x::Float64)
    normalizer = (sqrt(2 * pi) * N.sigma)^-1
    functional = exp( -((x - N.mu)^2) / (2*N.sigma ^2) ) 
    return (normalizer * functional)::Float64
end
function ppf(N::Gaussian,p::Float64)
    return N.mu - N.sigma * sqrt2  * erfcinv(2 * p)
end 
function trunc(N::Gaussian, margin::Float64, tie::Bool)
    #draw_margin = calc_draw_margin(draw_probability, size, self)
    _alpha = (-margin-N.mu)/N.sigma
    _beta  = ( margin-N.mu)/N.sigma
    if !tie
        #t= -_alpha
        v = pdf(N01,-_alpha) / cdf(N01,-_alpha)
        w = v * (v + (-_alpha))
    else
        v = (pdf(N01,_alpha)-pdf(N01,_beta))/(cdf(N01,_beta)-cdf(N01,_alpha))
        u = (_alpha*pdf(N01,_alpha)-_beta*pdf(N01,_beta))/(cdf(N01,_beta)-cdf(N01,_alpha))
        w =  - ( u - v^2 ) 
    end 
    mu = N.mu + N.sigma * v
    sigma = N.sigma*sqrt(1-w)
    return Gaussian(mu, sigma)
end
function Base.:+(N::Gaussian, M::Gaussian)
    mu = N.mu + M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma )
end
function Base.:-(N::Gaussian, M::Gaussian)
    mu = N.mu - M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma )
end
function Base.:*(N::Gaussian, M::Gaussian)
    _pi = N.pi + M.pi
    _tau = N.tau + M.tau
    return Gaussian(_tau/_pi, sqrt(1/_pi))        
end
function Base.:/(N::Gaussian, M::Gaussian)
    _pi = N.pi - M.pi
    _tau = N.tau - M.tau
    return Gaussian(_tau/_pi, sqrt(1/_pi))        
end
#
g1 = Gaussian(0.0, SIGMA)
g2 = Gaussian(2.0, 3.0)
@time g2/g1
#
function compute_margin(draw_probability::Float64,size::Int64)
    _N = Gaussian(0.0, sqrt(size)*BETA)
    res = abs(ppf(_N, 0.5-draw_probability/2))
    return res 
end



end # module

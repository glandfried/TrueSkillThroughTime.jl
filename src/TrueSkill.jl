module TrueSkill

#import SpecialFunctions

global const MU = 25.0::Float64
global const SIGMA = (MU/3)::Float64
global const BETA = (SIGMA / 2)::Float64
global const GAMMA = (SIGMA / 100)::Float64
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
global const Nms = Gaussian(MU, SIGMA)
global const N00 = Gaussian(0.0, 0.0)

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
function delta(N::Gaussian, M::Gaussian)
    return abs(N.mu - M.mu) , abs(N.sigma - M.sigma) 
end
function exclude(N::Gaussian,M::Gaussian)
    return Gaussian(N.mu - M.mu, sqrt(N.sigma^2 - M.sigma^2) )
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
function Base.isapprox(N::Gaussian, M::Gaussian, atol::Real=0)
    return (abs(N.mu - M.mu) < atol) & (abs(N.sigma - M.sigma) < atol)
end
function compute_margin(draw_probability::Float64,size::Int64)
    _N = Gaussian(0.0, sqrt(size)*BETA)
    res = abs(ppf(_N, 0.5-draw_probability/2))
    return res 
end
mutable struct Rating
    N::Gaussian
    beta::Float64
    gamma::Float64
    name::String
    function Rating()
        return new(Nms, BETA, GAMMA, "")
    end
    function Rating(mu::Float64, sigma::Float64)
        return new(Gaussian(mu, sigma), BETA, GAMMA, "")
    end
    function Rating(mu::Float64, sigma::Float64, beta::Float64)
        return new(Gaussian(mu, sigma), beta, GAMMA, "")
    end
    function Rating(N::Gaussian)
        return new(N, BETA, GAMMA, "")
    end
    function Rating(N::Gaussian,beta::Float64)
        return new(N, beta, GAMMA, "")
    end
    function Rating(N::Gaussian,beta::Float64,gamma::Float64)
        return new(N, beta, gamma, "")
    end
    function Rating(N::Gaussian,beta::Float64,name::String)
        return new(N, beta, GAMMA, name)
    end
    function Rating(N::Gaussian,beta::Float64,gamma::Float64,name::String)
        return new(N, beta, gamma, name)
    end
end

Base.show(io::IO, r::Rating) = print("Rating(", round(r.N.mu,digits=3)," ,", round(r.N.sigma,digits=3), ")")
function forget(R::Rating)
    _sigma = sqrt(R.N.sigma^2 + R.gamma^2)
    return _sigma
end 
function forget(R::Rating, t::Int64)
    _sigma = sqrt(R.N.sigma^2 + (R.gamma*t)^2)
    return _sigma
end 
function performance(R::Rating)
    _sigma = sqrt(R.N.sigma^2 + R.beta^2)
    return Gaussian(R.N.mu, _sigma)
end
mutable struct Game
    # Mutable?
    teams::Vector{Vector{Rating}}
    result::Vector{Int64}
    order::Vector{Int64}
    function Game(teams::Vector{Vector{Rating}}, result::Vector{Int64})
        if length(teams) != length(result)
            return error("length(teams) != length(result)")
        end
        return new(teams,result,sortperm(result))#,_infs, _infs)
    end
end
Base.length(G::Game) = length(G.result)
function size(G::Game)
    res = 0::Int64
    for e in 1:length(G)
        for i in 1:length(G.teams[e])
            res += 1
        end
    end
    return res
end
function performance(G::Game,i::Int64)
    res = N00
    for r in G.teams[i]
        res += performance(r)
    end
    return res
end 
mutable struct team_messages
    prior::Gaussian
    likelihood_lose::Gaussian
    likelihood_win::Gaussian
end
function p(tm::team_messages)
    return tm.prior*tm.likelihood_lose*tm.likelihood_win
end
function posterior_win(tm::team_messages)
    return tm.prior*tm.likelihood_lose
end
function posterior_lose(tm::team_messages)
    return tm.prior*tm.likelihood_win
end
function posterior(tm::team_messages)
    return tm.likelihood_win*tm.likelihood_lose
end
mutable struct diff_messages
    prior::Gaussian
    posterior::Gaussian
end
function p(dm::diff_messages)
    return dm.prior*dm.likelihood
end
function teams(G::Game)
    return [team_messages(performance(G,G.order[e]), Ninf, Ninf) for e in 1:length(G)]
end
function diffs(G::Game)
    return [diff_messages(Ninf,Ninf) for _ in 1:length(G)-1]
end
function update(dm::diff_messages,ta::team_messages,tb::team_messages,margin::Float64,tie::Bool)
    dm.prior = posterior_win(ta) - posterior_lose(tb)
    dm.posterior = trunc(dm.prior,margin,tie)/dm.prior
end
function Base.max(tuple1::Tuple{Float64,Float64}, tuple2::Tuple{Float64,Float64})
    return max(tuple1[1],tuple2[1]), max(tuple1[2],tuple2[2])
end
function Base.:>(tuple::Tuple{Float64,Float64}, threshold::Float64)
    return (tuple[1] > threshold) | (tuple[2] > threshold)
end
function posterior_teams(g::Game, margin::Float64)#margin=0.5
    o = g.order
    r = g.result
    t = teams(g)
    d = diffs(g)
    step = (Inf, Inf)::Tuple{Float64,Float64}
    iter = 0::Int64
    while (step > 1e-6) & (iter < 10)
        step = (0., 0.)
        for e in 1:length(d)-1
            update(d[e],t[o[e]],t[o[e+1]],margin,r[o[e]]==r[o[e+1]])
            likelihood_lose = (posterior_win(t[o[e]]) - d[e].posterior)
            step = max(step,delta(t[o[e+1]].likelihood_lose,likelihood_lose))
            t[o[e+1]].likelihood_lose = likelihood_lose
        end
        for e in length(d):-1:2
            update(d[e],t[o[e]],t[o[e+1]],margin,r[o[e]]==r[o[e+1]])
            likelihood_win = (posterior_lose(t[o[e+1]]) + d[e].posterior)
            step = max(step,delta(t[o[e]].likelihood_win,likelihood_win))
            t[o[e]].likelihood_win = likelihood_win
        end
        iter += 1
    end
    e = 1 
    update(d[e],t[o[e]],t[o[e+1]],margin,r[o[e]]==r[o[e+1]])
    t[o[e]].likelihood_win = (posterior_lose(t[o[e+1]]) + d[e].posterior)
    e = length(d) 
    update(d[e],t[o[e]],t[o[e+1]],margin,r[o[e]]==r[o[e+1]])
    t[o[e+1]].likelihood_lose = (posterior_win(t[o[e]]) - d[e].posterior)
    return [ posterior(t[e]) for e in 1:length(t)]
end
function posterior_performance(g::Game,margin::Float64)
    m_t_ft = posterior_teams(g,margin)
    return [[ m_t_ft[e] - exclude(performance(g,e),g.teams[e][i].N) for i in 1:length(g.teams[e])] for e in 1:length(g)]
end
function posteriors(g::Game,proba::Float64)
    margin = compute_margin(proba,size(G))
    m_p_fp = posterior_performance(g,margin)
    return [[ m_p_fp[e][i] * g.teams[e][i].N for i in 1:length(g.teams[e])] for e in 1:length(g)]
end


#post, setp, iter = posterior_skill(g,0.)
#g3 = Game([[Rating()],[Rating()],[Rating()]], [1,0,2])
#@time post3 = posterior_skill(g3,0.)
#post3[1][1].mu 
#post3[2][1]
#post3[3][1]

end # module

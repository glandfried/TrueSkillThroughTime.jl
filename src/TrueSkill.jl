module TrueSkill

#using Parameters
#import SpecialFunctions

global const MU = 25.0::Float64
global const SIGMA = (MU/3)::Float64
global const BETA = (SIGMA / 2)::Float64
global const GAMMA = (SIGMA / 100)::Float64
global const DRAW_PROBABILITY = 0.0::Float64
global const EPSILON = 1e-3::Float64
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
    function Gaussian(mu::Float64=MU,sigma::Float64=SIGMA)
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
global const N0g = Gaussian(0.0, GAMMA)
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
function ppf(N::Gaussian, p::Float64)
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
    _mu = 1/_pi == Inf ? 0. : (N.tau + M.tau)/_pi
    return Gaussian(_mu, sqrt(1/_pi))        
end
function Base.:/(N::Gaussian, M::Gaussian)
    _pi = N.pi - M.pi
    _mu = 1/_pi == Inf ? 0. : (N.tau - M.tau)/_pi
    return Gaussian(_mu, sqrt(1/_pi))        
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
    function Rating(mu::Float64=MU, sigma::Float64=SIGMA,beta::Float64=BETA,gamma::Float64=GAMMA,name::String="")
        return new(Gaussian(mu, sigma), beta, gamma, name)
    end
    function Rating(N::Gaussian,beta::Float64=BETA,gamma::Float64=GAMMA,name::String="")
        return new(N, beta, gamma, name)
    end
end
Base.show(io::IO, r::Rating) = print("Rating(", round(r.N.mu,digits=3)," ,", round(r.N.sigma,digits=3), ")")
Base.copy(r::Rating) = Rating(r.N,r.beta,r.gamma,r.name)
function forget(R::Rating, t::Float64)
    _sigma = max(sqrt(R.N.sigma^2 + (R.gamma*t)^2), SIGMA)
    return Gaussian(R.N.mu, _sigma)
end 
function performance(R::Rating)
    _sigma = sqrt(R.N.sigma^2 + R.beta^2)
    return Gaussian(R.N.mu, _sigma)
end
mutable struct Game
    # Mutable?
    teams::Vector{Vector{Rating}}
    result::Vector{Int64}
    margin::Float64
    likelihoods::Vector{Vector{Gaussian}}
    evidence::Float64
    function Game(teams::Vector{Vector{Rating}}, result::Vector{Int64},draw_proba::Float64=0.0)
        if length(teams) != length(result)
            return error("length(teams) != length(result)")
        end
        if (0.0 > draw_proba) | (1.0 <= draw_proba)
            return error("0.0 <= Draw probability < 1.0")
        elseif 0.0 == draw_proba
            margin = 0.0
        else
            margin = compute_margin(draw_proba,sum([ length(teams[e]) for e in 1:length(teams)]) )
        end
        _g = new(teams,result,margin,[],0.0)
        likelihoods(_g)
        return _g
    end
end        
Base.length(G::Game) = length(G.result)
#function Base.getindex
function size(G::Game)
    return [length(g.teams[e]) for e in 1:length(g.teams)]
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
function likelihood(tm::team_messages)
    return tm.likelihood_win*tm.likelihood_lose
end
mutable struct diff_messages
    prior::Gaussian
    likelihood::Gaussian
end
function p(dm::diff_messages)
    return dm.prior*dm.likelihood
end
function update(dm::diff_messages,ta::team_messages,tb::team_messages,margin::Float64,tie::Bool)
    dm.prior = posterior_win(ta) - posterior_lose(tb)
    dm.likelihood = trunc(dm.prior,margin,tie)/dm.prior
end
function Base.max(tuple1::Tuple{Float64,Float64}, tuple2::Tuple{Float64,Float64})
    return max(tuple1[1],tuple2[1]), max(tuple1[2],tuple2[2])
end
function Base.:>(tuple::Tuple{Float64,Float64}, threshold::Float64)
    return (tuple[1] > threshold) | (tuple[2] > threshold)
end
function likelihood_teams(g::Game)
    r = g.result
    o = sortperm(r)
    t = [team_messages(performance(g,o[e]), Ninf, Ninf) for e in 1:length(g)]
    d = [diff_messages(Ninf,Ninf) for _ in 1:length(g)-1]
    tie = [r[o[e]]==r[o[e+1]] for e in 1:length(d)]
    step = (Inf, Inf)::Tuple{Float64,Float64}
    iter = 0::Int64
    while (step > 1e-6) & (iter < 10)
        step = (0., 0.)
        for e in 1:length(d)-1
            d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
            d[e].likelihood = trunc(d[e].prior,g.margin,tie[e])/d[e].prior
            likelihood_lose = (posterior_win(t[e]) - d[e].likelihood)
            step = max(step,delta(t[e+1].likelihood_lose,likelihood_lose))
            t[e+1].likelihood_lose = likelihood_lose
        end
        for e in length(d):-1:2
            d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
            d[e].likelihood = trunc(d[e].prior,g.margin,tie[e])/d[e].prior
            likelihood_win = (posterior_lose(t[e+1]) + d[e].likelihood)
            step = max(step,delta(t[e].likelihood_win,likelihood_win))
            t[e].likelihood_win = likelihood_win
        end
        iter += 1
    end
    e = 1 
    d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
    d[e].likelihood = trunc(d[e].prior,g.margin,tie[e])/d[e].prior
    t[e].likelihood_win = (posterior_lose(t[e+1]) + d[e].likelihood)
    e = length(d) 
    d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
    d[e].likelihood = trunc(d[e].prior,g.margin,tie[e])/d[e].prior
    t[e+1].likelihood_lose = (posterior_win(t[e]) - d[e].likelihood)
    
    g.evidence = prod([1-cdf(d[e].prior,0.0) for e in 1:length(d)])
    
    return [ likelihood(t[o[e]]) for e in 1:length(t)] 
end
function likelihoods(g::Game)
    m_t_ft = likelihood_teams(g)
    g.likelihoods = [[ m_t_ft[e] - exclude(performance(g,e),g.teams[e][i].N) for i in 1:length(g.teams[e])] for e in 1:length(g)]
    return g.likelihoods
end
function posteriors(g::Game)
    return [[ g.likelihoods[e][i] * g.teams[e][i].N for i in 1:length(g.teams[e])] for e in 1:length(g)]
end
function update(g::Game, priors::Array{Array{Gaussian,1},1})
    for e in 1:length(g.teams)
        for i in length(g.teams[e])
            g.teams[e][i].N = priors[e][i]
        end
    end
    likelihoods(g)
end
mutable struct Batch
    events::Vector{Vector{Vector{String}}}
    results::Vector{Vector{Int64}}
    time::Float64
    elapsed::Dict{String,Float64}
    prior_forward::Dict{String,Rating}
    prior_backward::Dict{String,Gaussian}
    likelihood::Dict{String,Dict{Int64,Gaussian}}
    evidences::Vector{Float64}
    partake::Dict{String,Vector{Int64}}
    agents::Set{String}
    function Batch(events::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}} 
                 ,time::Float64, last_time::Dict{String,Float64}=Dict{String,Float64}() , priors::Dict{String,Rating}=Dict{String,Rating}())
        if length(events)!= length(results)
            error("length(events)!= length(results)")
        end
        b = new(events, results, time, last_time, priors
                   ,Dict{String,Gaussian}()
                   ,Dict{String,Dict{Int64,Gaussian}}()
                   ,[0.0 for _ in 1:length(events)]
                   ,Dict{String,Vector{Int64}}()
                   ,Set{String}())
        
        b.agents = Set(vcat((b.events...)...))
        for a in b.agents#a="c"
            b.partake[a] = [e for e in 1:length(b.events) for team in b.events[e] if a in team ]
            b.elapsed[a] = haskey(last_time, a) ? (time - last_time[a]) : 0.0
            if !haskey(priors, a)
                b.prior_forward[a] = Rating(Nms,BETA,GAMMA,a)
            else 
                forget(b.prior_forward[a],b.elapsed)
            end
            b.prior_backward[a] = Ninf
            b.likelihood[a] = Dict{Int64,Gaussian}()
            for e in b.partake[a]
                b.likelihood[a][e] = Ninf
            end
        end
        iteration(b)
        return b
    end
end

Base.show(io::IO, b::Batch) = print("Batch(time=", b.time, ", events=", b.events, ", results=", b.results,")")
Base.length(b::Batch) = length(b.results)
function likelihood(b::Batch, agent::String)   
    return prod([value for (_, value) in b.likelihood[agent]])
end
function posterior(b::Batch, agent::String)
    return likelihood(b, agent)*b.prior_backward[agent]*b.prior_forward[agent].N   
end
function within_prior(b::Batch, agent::String, event::Int64)
    res = copy(b.prior_forward[agent])
    res.N = posterior(b,agent)/b.likelihood[agent][event]
    return res
end
function within_priors(b::Batch, event::Int64)
    return [[within_prior(b, a, event) for a in team] for team in b.events[event]]
end
function iteration(b::Batch)
    for e in 1:length(b)
        g = Game(within_priors(b,e), b.results[e])
        teams = b.events[e]
        for t in 1:length(teams)
            for j in 1:length(teams[t])
                b.likelihood[teams[t][j]][e] = g.likelihoods[t][j] 
            end
        end
        b.evidences[e] = g.evidence
    end
end
function posterior_forward(b::Batch, agent::String)
    res = copy(b.prior_forward[agent])
    res.N = b.priors_forward[a]*likelihood(b,a)
    return res
end
function posterior_backward(b::Batch, agent::String)
    res = copy(b.prior_forward[agent])
    res.N = likelihood(b,a)*b.prior_backward[a]
    return res
end
function forward_priors_out(b::Batch)
    res = Dict{String,Rating}()
    for a in b.agents
        res[a] = posterior_forward(b,a)
    end
    return res
end
function backward_priors_out(b::Batch)
    res = Dict{String,Rating}()
    for a in b.agents
        res[a] = forget(posterior_backward(b,a),b.elapsed[a])
    end
    return res
end

function convergence(b::Batch)
    step = (Inf, Inf)::Tuple{Float64,Float64}
    iter = 0::Int64
    while (step > 1e-3) & (iter < 10)
        step = (0., 0.)
        old_likelihood = deepcopy(b.likelihood)
        iteration(b)
        for (a, dict) in b.likelihood
            for (e, value) in dict
                step = max(step,delta(value,old_likelihood[a][e]))               
            end
        end
        iter += 1
    end
    return step , iter
end

# 

# old_likelihood = [ [b.likelihood[a][1] for a in teams] for teams in b.events[1]]
# b.likelihood["a"][1]
# b.likelihood["a"][1] = N01
# old_likelihood[1][1] 
# 
# g = Game([[Rating()],[Rating()]], [0,1])
# posteriors(g)
# g2 = Game([[Rating(29.205,1.)],[Rating()]], [1,0])         
# posteriors(g2 )[2][1]
# @time b = Batch([ [["a"],["b"]], [["a"],["c"]] ], [[0,1],[1,0]],2.)
# 
# b.events
# b.likelihood
# b.evidences
# 
# iteration(b)
# @time within_priors(b,2)
# 
# @time g3 = Game([[Rating()],[Rating()]], [1,0])
# @time g3.evidence
# @time update(g3,posteriors(g3) )
# @time g3.evidence
# @time update(g3,posteriors(g3) )
# @time g3.evidence
# posteriors(g3)

#post, setp, iter = posterior_skill(g,0.)
#g3 = Game([[Rating()],[Rating()],[Rating()]], [1,0,2])
#@time post3 = posterior_skill(g3,0.)
#post3[1][1].mu 
#post3[2][1]
#post3[3][1]

end # module
